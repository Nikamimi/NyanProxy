"""
Anthropic Route Handler for NyanProxy

Handles Anthropic API proxying with retry logic, token validation, and logging
Extracted from core/app.py for better modularity
"""
import os
import time
import json
import hashlib
from typing import Dict, Any, Tuple
from flask import request, jsonify, Response, g

from .base_handler import BaseHandler
from src.services.model_families import AIProvider
from src.middleware.auth import check_quota, track_token_usage
from ai.tokenizers.unified_tokenizer import unified_tokenizer


class AnthropicHandler(BaseHandler):
    """Handler for Anthropic API routes"""
    
    def messages(self) -> Tuple[Any, int]:
        """Handle Anthropic messages with retry logic and validation"""
        start_time = self.track_request_start()
        max_retries = int(os.getenv('MAX_RETRIES', '3'))
        
        try:
            # Check quota for Anthropic models
            has_quota, quota_error = check_quota('anthropic')
            if not has_quota:
                self.track_request_end('chat_completions', start_time, error=True)
                return self.create_error_response(quota_error, 429)
            
            # Validate model is whitelisted
            request_json = request.get_json() if request else {}
            model = request_json.get('model', 'claude-3-haiku-20240307')
            
            if not self.model_manager.is_model_whitelisted(AIProvider.ANTHROPIC, model):
                self.track_request_end('chat_completions', start_time, error=True)
                return self.create_error_response({
                    "message": f"Model '{model}' is not whitelisted for use",
                    "type": "model_not_allowed"
                }, 403)
            
            # Validate token limits
            token_validation_error = self._validate_token_limits(request_json, model, start_time)
            if token_validation_error:
                return token_validation_error
            
            # Retry logic with different keys
            return self._execute_with_retry(request_json, model, max_retries, start_time)
            
        except Exception as e:
            self.track_request_end('chat_completions', start_time, error=True)
            return self.create_error_response(str(e), 500)
    
    def models(self) -> Tuple[Any, int]:
        """Anthropic models endpoint (placeholder - not yet implemented)"""
        start_time = self.track_request_start()
        
        try:
            self.track_request_end('models', start_time, error=True)
            return self.create_error_response("Anthropic models endpoint not implemented yet", 501)
            
        except Exception as e:
            self.track_request_end('models', start_time, error=True)
            return self.create_error_response(str(e), 500)
    
    def _validate_token_limits(self, request_json: Dict, model: str, start_time: float) -> Tuple[Any, int]:
        """Validate token limits for the model"""
        model_info = self.model_manager.get_model_info(model)
        if not model_info or not (model_info.max_input_tokens or model_info.max_output_tokens):
            return None
        
        print(f"Token validation for model {model}: max_input={model_info.max_input_tokens}, max_output={model_info.max_output_tokens}")
        
        # Count input tokens using the unified tokenizer
        input_tokens = 0
        try:
            token_result = unified_tokenizer.count_tokens(request_json, 'anthropic')
            input_tokens = token_result.get('input_tokens', 0) or token_result.get('prompt_tokens', 0)
            print(f"Unified tokenizer result: {input_tokens} tokens")
        except Exception as token_error:
            print(f"Unified tokenizer failed: {token_error}")
        
        # Fallback token counting if unified tokenizer returned 0
        if input_tokens == 0:
            print("Unified tokenizer returned 0, using character estimation...")
            messages = request_json.get('messages', [])
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            input_tokens = max(1, total_chars // 4)  # Conservative 4 chars per token
            print(f"Character fallback: {input_tokens} tokens (from {total_chars} chars)")
        
        print(f"Final input token count: {input_tokens}")
        
        # Check output tokens limit (from max_tokens for Anthropic)
        max_completion_tokens = request_json.get('max_tokens')
        print(f"Requested output tokens: {max_completion_tokens}")
        
        # Validate token limits
        is_valid, error_message = model_info.validate_token_limits(input_tokens, max_completion_tokens)
        print(f"Validation result: {is_valid}, message: {error_message}")
        
        if not is_valid:
            print("Token limit exceeded, returning error")
            
            # Log token limit violation
            self._log_token_violation(model, input_tokens, max_completion_tokens, model_info, error_message)
            
            self.track_request_end('chat_completions', start_time, error=True)
            return self.create_error_response({
                "message": error_message,
                "type": "token_limit_exceeded"
            }, 400)
        
        return None
    
    def _log_token_violation(self, model: str, input_tokens: int, max_completion_tokens: int, 
                           model_info: Any, error_message: str):
        """Log token limit violation"""
        auth_data = self.get_auth_data()
        if auth_data.get("type") == "user_token":
            user_token = auth_data["token"]
            user = auth_data["user"]
            client_ip = self.get_client_ip()
            ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
            user_agent = request.headers.get('User-Agent', '')
            
            # Record token violation for happiness tracking
            if hasattr(user, 'record_token_violation'):
                user.record_token_violation()
            
            # Log the violation event
            self.structured_logger.log_user_action(
                user_token=user_token,
                action='token_limit_exceeded',
                details={
                    'model': model,
                    'input_tokens': input_tokens,
                    'max_input_tokens': model_info.max_input_tokens,
                    'requested_output_tokens': max_completion_tokens,
                    'max_output_tokens': model_info.max_output_tokens,
                    'error_message': error_message,
                    'ip_hash': ip_hash,
                    'user_agent': user_agent
                }
            )
    
    def _execute_with_retry(self, request_json: Dict, model: str, max_retries: int, start_time: float) -> Tuple[Any, int]:
        """Execute API call with retry logic using different keys"""
        used_keys = []
        
        for attempt in range(max_retries):
            # Get API key, excluding previously used keys
            exclude_key = used_keys[-1] if used_keys else None
            api_key = self.key_manager.get_api_key('anthropic', exclude_key=exclude_key)
            
            if not api_key:
                print(f"No API keys available for Anthropic on attempt {attempt + 1}")
                self.track_request_end('chat_completions', start_time, error=True)
                return self.create_error_response("No Anthropic API key configured", 500)
            
            used_keys.append(api_key)
            print(f"Attempt {attempt + 1}/{max_retries} using key {api_key[:8]}...{api_key[-4:]}")
            
            # Make API call
            result = self._make_api_call(api_key, request_json, model, start_time, attempt, max_retries, used_keys)
            if result:
                return result
            
            # If we reach here, continue to next retry
            time.sleep(1)  # Brief pause before retry
        
        # All retries failed
        self.track_request_end('chat_completions', start_time, error=True)
        return self.create_error_response("All API keys rate limited or failed", 429)
    
    def _make_api_call(self, api_key: str, request_json: Dict, model: str, start_time: float, 
                      attempt: int, max_retries: int, used_keys: list) -> Tuple[Any, int]:
        """Make actual API call to Anthropic"""
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        try:
            session = self.connection_pool.get_session('anthropic')
            response = session.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=request_json,
                stream=request_json.get('stream', False),
                timeout=getattr(session, '_default_timeout', (10, 30))
            )
            
            response_time = time.time() - start_time
            
            # Handle error responses
            if response.status_code >= 400:
                error_text = response.text
                if self.key_manager.handle_api_error('anthropic', api_key, error_text, response.status_code):
                    if attempt < max_retries - 1:
                        return None  # Continue to next retry
                
                # Final error - return the error response
                self.track_request_end('chat_completions', start_time, error=True)
                try:
                    error_data = json.loads(error_text)
                    return Response(json.dumps(error_data), status=response.status_code, 
                                  content_type='application/json'), response.status_code
                except:
                    return self.create_error_response({
                        "message": error_text,
                        "type": "anthropic_api_error"
                    }, response.status_code)
            
            # Success - update key health
            print(f"Anthropic API call successful on attempt {attempt + 1}")
            print(f"Keys tried: {[k[:8] + '...' + k[-4:] for k in used_keys]}")
            self.key_manager.update_key_health(api_key, True)
            
            # Extract and track tokens
            tokens = self._extract_tokens(response, request_json, model)
            
            self.track_request_end('chat_completions', start_time, error=False, tokens=tokens, service='anthropic')
            
            # Log completion events
            if tokens:
                model_cost = 0.0
                auth_data = self.get_auth_data()
                if auth_data.get('type') == 'user_token':
                    # Track model usage and get cost
                    model_cost = self.model_manager.track_model_usage(
                        user_token=auth_data['token'],
                        model_id=model,
                        input_tokens=tokens.get('prompt_tokens', 0),
                        output_tokens=tokens.get('completion_tokens', 0),
                        success=True
                    )
                    
                    # Track token usage
                    track_token_usage(
                        model,
                        tokens.get('prompt_tokens', 0),
                        tokens.get('completion_tokens', 0),
                        cost=model_cost or 0.0,
                        response_time_ms=response_time * 1000
                    )
                
                # Log completion event
                self.log_completion_event('anthropic', model, tokens, model_cost or 0.0, response_time, True)
            
            # Return streaming or regular response
            is_streaming = request_json.get('stream', False)
            if is_streaming:
                return Response(
                    response.iter_content(chunk_size=1024),
                    content_type=response.headers.get('content-type'),
                    status=response.status_code
                ), response.status_code
            else:
                return Response(
                    response.content,
                    content_type=response.headers.get('content-type'),
                    status=response.status_code
                ), response.status_code
                
        except Exception as e:
            error_msg = str(e)
            if self.key_manager.handle_api_error('anthropic', api_key, error_msg):
                if attempt < max_retries - 1:
                    return None  # Continue to next retry
            
            self.track_request_end('chat_completions', start_time, error=True)
            return self.create_error_response(error_msg, 500)
    
    def _extract_tokens(self, response: Any, request_json: Dict, model: str) -> Dict[str, int]:
        """Extract token information from Anthropic response"""
        try:
            response_content = response.content
            is_streaming = request_json.get('stream', False)
            
            if response.status_code == 200:
                try:
                    response_data = json.loads(response_content)
                    
                    # First try to get tokens from API response (usage field)
                    if 'usage' in response_data:
                        usage = response_data['usage']
                        tokens = {
                            'prompt_tokens': usage.get('input_tokens', 0),
                            'completion_tokens': usage.get('output_tokens', 0),
                            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                        }
                        print(f"Token tracking for {model}: input={tokens['prompt_tokens']}, output={tokens['completion_tokens']} (API provided)")
                        return tokens
                    else:
                        # Fallback: use tokenizer to estimate
                        response_text = ""
                        if 'content' in response_data:
                            for content in response_data['content']:
                                if content.get('type') == 'text':
                                    response_text += content.get('text', '') + " "
                        
                        # Use tokenizer to count tokens
                        token_result = unified_tokenizer.count_tokens(
                            request_data=request_json,
                            service="anthropic",
                            response_text=response_text.strip() if response_text else None
                        )
                        tokens = {
                            'prompt_tokens': token_result['prompt_tokens'],
                            'completion_tokens': token_result['completion_tokens'],
                            'total_tokens': token_result['total_tokens']
                        }
                        print(f"Token tracking for {model}: input={tokens['prompt_tokens']}, output={tokens['completion_tokens']} (unified tokenizer)")
                        return tokens
                        
                except Exception as e:
                    # Fallback: count request tokens at minimum
                    try:
                        # Extract text from request for token counting
                        request_text = ""
                        if 'messages' in request_json:
                            for msg in request_json['messages']:
                                if 'content' in msg:
                                    request_text += str(msg['content']) + " "
                        if 'system' in request_json:
                            request_text += str(request_json['system']) + " "
                        
                        # Use simple character-based estimation if tokenizer fails
                        estimated_prompt_tokens = max(len(request_text) // 4, 100)  # ~4 chars per token
                        estimated_completion_tokens = 50  # Default completion estimate
                        
                        tokens = {
                            'prompt_tokens': estimated_prompt_tokens,
                            'completion_tokens': estimated_completion_tokens,
                            'total_tokens': estimated_prompt_tokens + estimated_completion_tokens
                        }
                        print(f"Token tracking for {model}: input={tokens['prompt_tokens']}, output={tokens['completion_tokens']} (character fallback)")
                        return tokens
                    except Exception:
                        # Absolute fallback
                        tokens = {
                            'prompt_tokens': 1000,  # Higher default for large requests
                            'completion_tokens': 100,
                            'total_tokens': 1100
                        }
                        print(f"Token tracking for {model}: input={tokens['prompt_tokens']}, output={tokens['completion_tokens']} (absolute fallback)")
                        return tokens
        
        except Exception as e:
            print(f"Error in token extraction: {e}")
            return {
                'prompt_tokens': 500,
                'completion_tokens': 50,
                'total_tokens': 550
            }