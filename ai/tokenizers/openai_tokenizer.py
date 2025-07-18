"""OpenAI tokenizer with proper chat formatting and model-specific logic."""

import time
from typing import List, Dict, Any, Union, Optional
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class OpenAITokenizer:
    """Accurate OpenAI tokenizer with proper message formatting."""
    
    def __init__(self):
        self.encoder = None
        self.gpt4_vision_system_prompt_size = 170
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except:
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                except:
                    pass
    
    def _get_encoder_for_model(self, model: str):
        """Get the appropriate encoder for a model, with fallback for unknown models."""
        if not TIKTOKEN_AVAILABLE:
            return None
            
        try:
            # Try to get model-specific encoder
            return tiktoken.encoding_for_model(model)
        except:
            # Fallback to default encoder if model is unknown
            return self.encoder
    
    def count_tokens(self, prompt: Union[str, List[Dict[str, Any]]], model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Count tokens for OpenAI request with proper formatting."""
        start_time = time.time()
        
        if isinstance(prompt, str):
            result = self._count_text_tokens(prompt)
        else:
            result = self._count_chat_tokens(prompt, model)
        
        result["tokenization_duration_ms"] = (time.time() - start_time) * 1000
        return result
    
    def _count_text_tokens(self, text: str) -> Dict[str, Any]:
        """Count tokens for plain text."""
        if len(text) > 500000:
            return {
                "tokenizer": "length fallback",
                "token_count": 100000,
            }
        
        if not self.encoder:
            # Fallback: approximate token count (1 token ≈ 4 chars)
            return {
                "tokenizer": "character fallback",
                "token_count": len(text) // 4,
            }
        
        return {
            "tokenizer": "tiktoken",
            "token_count": len(self.encoder.encode(text)),
        }
    
    def _count_chat_tokens(self, messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
        """Count tokens for chat messages with proper OpenAI formatting."""
        # Get model-specific encoder with fallback
        encoder = self._get_encoder_for_model(model)
        
        if not encoder:
            # Fallback for chat messages
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return {
                "tokenizer": "character fallback",
                "token_count": total_chars // 4,
            }
        
        # Model-specific formatting
        old_formatting = model.startswith("turbo-0301")
        vision = "vision" in model
        
        tokens_per_message = 4 if old_formatting else 3
        tokens_per_name = -1 if old_formatting else 1
        
        num_tokens = self.gpt4_vision_system_prompt_size if vision else 0
        
        for message in messages:
            num_tokens += tokens_per_message
            
            for key, value in message.items():
                if not value:
                    continue
                
                text_content = ""
                
                if isinstance(value, list):
                    # Handle multimodal content
                    for item in value:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_content += item.get("text", "")
                            elif item.get("type") in ["image", "image_url"]:
                                # For images, add a base cost (would need image processing for exact)
                                num_tokens += 85  # Base cost for images
                else:
                    text_content = str(value)
                
                if len(text_content) > 800000 or num_tokens > 200000:
                    raise ValueError("Content is too large to tokenize.")
                
                try:
                    num_tokens += len(encoder.encode(text_content))
                except Exception as e:
                    # Fallback to character count if tokenization fails
                    num_tokens += len(text_content) // 4
                
                if key == "name":
                    num_tokens += tokens_per_name
        
        # Every reply is primed with assistant message
        num_tokens += 3
        
        return {
            "tokenizer": "tiktoken",
            "token_count": num_tokens,
        }
    
    def extract_text_from_request(self, request_data: Dict[str, Any]) -> str:
        """Extract text from OpenAI request for token counting."""
        messages = request_data.get("messages", [])
        text_parts = []
        
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
        
        return " ".join(text_parts)