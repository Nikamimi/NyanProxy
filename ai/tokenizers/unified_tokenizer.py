"""Unified tokenizer interface for accurate token counting across different AI services."""

from typing import Dict, Any, Union, Optional
from .openai_tokenizer import OpenAITokenizer
from .anthropic_tokenizer import AnthropicTokenizer


class UnifiedTokenizer:
    """Unified interface for token counting across different AI services."""
    
    def __init__(self):
        self.openai_tokenizer = OpenAITokenizer()
        self.anthropic_tokenizer = AnthropicTokenizer()
    
    def count_tokens(
        self, 
        request_data: Dict[str, Any], 
        service: str, 
        model: str = None,
        response_text: str = None
    ) -> Dict[str, Any]:
        """Count tokens for a request with proper service-specific formatting."""
        
        if service == "openai":
            return self._count_openai_tokens(request_data, model, response_text)
        elif service == "anthropic":
            return self._count_anthropic_tokens(request_data, response_text)
        else:
            # Fallback for unknown services
            return self._count_fallback_tokens(request_data, response_text)
    
    def _count_openai_tokens(self, request_data: Dict[str, Any], model: str = None, response_text: str = None) -> Dict[str, Any]:
        """Count OpenAI tokens with proper chat formatting."""
        model = model or "gpt-3.5-turbo"
        
        # Count prompt tokens
        messages = request_data.get("messages", [])
        prompt_result = self.openai_tokenizer.count_tokens(messages, model)
        prompt_tokens = prompt_result["token_count"]
        
        # Count completion tokens if response provided
        completion_tokens = 0
        if response_text:
            completion_result = self.openai_tokenizer.count_tokens(response_text, model)
            completion_tokens = completion_result["token_count"]
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokenizer": prompt_result["tokenizer"],
            "tokenization_duration_ms": prompt_result.get("tokenization_duration_ms", 0)
        }
    
    def _count_anthropic_tokens(self, request_data: Dict[str, Any], response_text: str = None) -> Dict[str, Any]:
        """Count Anthropic tokens with proper message formatting."""
        
        # Count prompt tokens (includes system + messages)
        prompt_result = self.anthropic_tokenizer.count_tokens(request_data)
        prompt_tokens = prompt_result["token_count"]
        
        # Count completion tokens if response provided
        completion_tokens = 0
        if response_text:
            completion_result = self.anthropic_tokenizer.count_tokens(response_text)
            completion_tokens = completion_result["token_count"]
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokenizer": prompt_result["tokenizer"],
            "tokenization_duration_ms": prompt_result.get("tokenization_duration_ms", 0)
        }
    
    def _count_fallback_tokens(self, request_data: Dict[str, Any], response_text: str = None) -> Dict[str, Any]:
        """Fallback token counting for unknown services."""
        # Extract text from request
        text_parts = []
        
        # Try common request formats
        if "messages" in request_data:
            for msg in request_data["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        text_parts.append(content)
        
        if "system" in request_data:
            text_parts.append(str(request_data["system"]))
        
        if "prompt" in request_data:
            text_parts.append(str(request_data["prompt"]))
        
        request_text = " ".join(text_parts)
        
        # Simple character-based estimation
        prompt_tokens = len(request_text) // 4  # ~4 chars per token
        completion_tokens = len(response_text) // 4 if response_text else 0
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokenizer": "character fallback",
            "tokenization_duration_ms": 0
        }
    
    def extract_text_from_request(self, request_data: Dict[str, Any], service: str) -> str:
        """Extract text from request for token counting."""
        if service == "openai":
            return self.openai_tokenizer.extract_text_from_request(request_data)
        elif service == "anthropic":
            return self.anthropic_tokenizer.extract_text_from_request(request_data)
        else:
            # Fallback text extraction
            text_parts = []
            if "messages" in request_data:
                for msg in request_data["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        text_parts.append(str(msg["content"]))
            return " ".join(text_parts)


# Global tokenizer instance
unified_tokenizer = UnifiedTokenizer()