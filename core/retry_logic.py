"""
Universal Retry Logic System for AI API Calls with Key Pool Management
"""
import os
import time
import logging
from typing import Callable, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback

from .key_pool_manager import KeyPoolManager
from .health_checker import HealthCheckManager


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    response: Any = None
    error_message: str = None
    attempts_made: int = 0
    keys_tried: list = None
    final_error_code: Optional[int] = None
    total_time: float = 0.0
    
    def __post_init__(self):
        if self.keys_tried is None:
            self.keys_tried = []


class UniversalRetrySystem:
    """
    Universal retry system that works with any AI provider
    Handles key rotation, error classification, and real-time pool updates
    """
    
    def __init__(self, key_pool_manager: KeyPoolManager):
        self.key_pool = key_pool_manager
        self.health_manager = HealthCheckManager()
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        
        # Error classification mappings for each provider
        self.error_classifications = {
            'openai': {
                401: 'invalid_key',
                429: 'quota_exceeded'
            },
            'anthropic': {
                401: 'invalid_key',
                429: 'rate_limited',
                403: 'quota_exceeded'
            },
            'google': {
                403: 'invalid_key',
                429: 'rate_limited'
            }
        }
        
        print(f"[RETRY] Universal Retry System initialized with MAX_RETRIES={self.max_retries}")
    
    def execute_with_retry(self, 
                          provider: str,
                          api_call_func: Callable,
                          *args,
                          **kwargs) -> RetryResult:
        """
        Execute an API call with automatic retry logic and key rotation
        
        Args:
            provider: AI provider name (e.g., 'openai', 'anthropic', 'google')
            api_call_func: Function that makes the API call
            *args, **kwargs: Arguments to pass to the API call function
            
        Returns:
            RetryResult containing success status, response, and metadata
        """
        start_time = time.time()
        attempts_made = 0
        keys_tried = []
        last_error = None
        last_error_code = None
        
        print(f"[RETRY] Starting retry execution for {provider} (max retries: {self.max_retries})")
        
        while attempts_made < self.max_retries:
            attempts_made += 1
            
            # Get a healthy key from the pool
            api_key_obj = self.key_pool.get_healthy_key(provider)
            if not api_key_obj:
                error_msg = f"No healthy API keys available for {provider} after {attempts_made-1} attempts"
                print(f"[FAIL] {error_msg}")
                return RetryResult(
                    success=False,
                    error_message=error_msg,
                    attempts_made=attempts_made,
                    keys_tried=keys_tried,
                    total_time=time.time() - start_time
                )
            
            api_key = api_key_obj.key
            keys_tried.append(api_key_obj.masked_key)
            
            print(f"[RETRY] Attempt {attempts_made}/{self.max_retries} using key {api_key_obj.masked_key}")
            
            try:
                # Execute the API call
                call_start_time = time.time()
                
                # Inject the API key into the function call
                # This assumes the function accepts an 'api_key' parameter
                if 'api_key' not in kwargs:
                    kwargs['api_key'] = api_key
                
                response = api_call_func(*args, **kwargs)
                call_time = time.time() - call_start_time
                
                # Mark successful usage
                self.key_pool.mark_key_result(
                    provider=provider,
                    key=api_key,
                    success=True,
                    response_time=call_time
                )
                
                print(f"[OK] API call successful on attempt {attempts_made} ({call_time:.2f}s)")
                return RetryResult(
                    success=True,
                    response=response,
                    attempts_made=attempts_made,
                    keys_tried=keys_tried,
                    total_time=time.time() - start_time
                )
                
            except Exception as e:
                call_time = time.time() - call_start_time
                error_status, should_retry = self._classify_error(provider, e)
                
                print(f"[FAIL] Attempt {attempts_made} failed: {error_status} - {str(e)}")
                
                # Mark key result based on error
                self.key_pool.mark_key_result(
                    provider=provider,
                    key=api_key,
                    success=False,
                    error_status=error_status,
                    error_message=str(e),
                    response_time=call_time
                )
                
                last_error = str(e)
                last_error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                
                # Check if we should continue retrying
                if not should_retry:
                    print(f"[STOP] Error type '{error_status}' is not retryable, stopping attempts")
                    break
                
                if attempts_made < self.max_retries:
                    # Small delay before next attempt
                    retry_delay = min(2 ** (attempts_made - 1), 10)  # Exponential backoff, max 10s
                    print(f" Waiting {retry_delay}s before next attempt...")
                    time.sleep(retry_delay)
        
        # All attempts exhausted
        total_time = time.time() - start_time
        print(f" All {attempts_made} attempts failed for {provider} in {total_time:.2f}s")
        print(f" Final error: {last_error}")
        print(f" Keys tried: {', '.join(keys_tried)}")
        
        return RetryResult(
            success=False,
            error_message=last_error,
            attempts_made=attempts_made,
            keys_tried=keys_tried,
            final_error_code=last_error_code,
            total_time=total_time
        )
    
    def _classify_error(self, provider: str, exception: Exception) -> Tuple[str, bool]:
        """
        Classify an error and determine if it's retryable
        
        Returns:
            Tuple of (error_status, should_retry)
        """
        # Try to extract status code from exception
        status_code = None
        error_message = str(exception)
        
        # Common ways exceptions store status codes
        if hasattr(exception, 'status_code'):
            status_code = exception.status_code
        elif hasattr(exception, 'code'):
            status_code = exception.code
        elif hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            status_code = exception.response.status_code
        
        # Use provider-specific classification if we have a status code
        if status_code and provider in self.error_classifications:
            provider_codes = self.error_classifications[provider]
            if status_code in provider_codes:
                error_type = provider_codes[status_code]
                # Don't retry invalid keys or quota exceeded
                should_retry = error_type not in ['invalid_key', 'quota_exceeded']
                return error_type, should_retry
        
        # Use health checker for more detailed classification
        try:
            status, is_retryable = self.health_manager.checkers[provider].classify_error(
                status_code or 0, error_message
            )
            return status, is_retryable
        except (KeyError, AttributeError):
            pass
        
        # Fallback classification based on common patterns
        error_lower = error_message.lower()
        
        # Network/connection errors - always retryable
        if any(term in error_lower for term in ['timeout', 'connection', 'network', 'dns']):
            return 'network_error', True
        
        # Rate limiting - retryable
        if any(term in error_lower for term in ['rate limit', 'too many requests']):
            return 'rate_limited', True
        
        # Authentication errors - not retryable
        if any(term in error_lower for term in ['unauthorized', 'invalid key', 'authentication']):
            return 'invalid_key', False
        
        # Quota errors - not retryable
        if any(term in error_lower for term in ['quota', 'billing', 'exceeded']):
            return 'quota_exceeded', False
        
        # Server errors - retryable
        if status_code and status_code >= 500:
            return 'server_error', True
        
        # Default: treat as temporary error that can be retried
        return 'unknown_error', True
    
    def execute_openai_call(self, call_func: Callable, *args, **kwargs) -> RetryResult:
        """Helper method for OpenAI API calls"""
        return self.execute_with_retry('openai', call_func, *args, **kwargs)
    
    def execute_anthropic_call(self, call_func: Callable, *args, **kwargs) -> RetryResult:
        """Helper method for Anthropic API calls"""
        return self.execute_with_retry('anthropic', call_func, *args, **kwargs)
    
    def execute_google_call(self, call_func: Callable, *args, **kwargs) -> RetryResult:
        """Helper method for Google API calls"""
        return self.execute_with_retry('google', call_func, *args, **kwargs)
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get statistics about retry operations"""
        return {
            'max_retries': self.max_retries,
            'key_pool_status': self.key_pool.get_pool_status(),
            'removed_keys': self.key_pool.get_removed_keys()
        }
    
    def set_max_retries(self, retries: int):
        """Update max retries configuration"""
        self.max_retries = max(1, retries)
        self.key_pool.set_max_retries(retries)
        print(f"[TOOL] Max retries updated to {self.max_retries}")


# Example usage functions for each provider
def example_openai_api_call(message: str, model: str = "gpt-4o-mini", api_key: str = None) -> Dict:
    """Example OpenAI API call function"""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=100
    )
    
    return {
        "content": response.choices[0].message.content,
        "usage": response.usage.dict() if response.usage else {}
    }


def example_anthropic_api_call(message: str, model: str = "claude-3-5-haiku-20241022", api_key: str = None) -> Dict:
    """Example Anthropic API call function"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": message}]
    )
    
    return {
        "content": response.content[0].text if response.content else "",
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }


def example_google_api_call(message: str, model: str = "gemini-1.5-flash", api_key: str = None) -> Dict:
    """Example Google API call function"""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(message)
    
    return {
        "content": response.text,
        "usage": {}  # Google doesn't provide detailed usage in basic calls
    }


# Example usage
if __name__ == "__main__":
    # This would be used in your main application
    key_pool = KeyPoolManager()
    retry_system = UniversalRetrySystem(key_pool)
    
    # Add some test keys
    key_pool.add_keys('openai', ['sk-test1', 'sk-test2'])
    key_pool.add_keys('anthropic', ['sk-ant-test1', 'sk-ant-test2'])
    
    # Example API call with retry
    result = retry_system.execute_openai_call(
        example_openai_api_call,
        message="Hello, world!",
        model="gpt-4o-mini"
    )
    
    if result.success:
        print(f"Success: {result.response}")
    else:
        print(f"Failed after {result.attempts_made} attempts: {result.error_message}")