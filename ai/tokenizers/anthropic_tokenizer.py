"""Anthropic tokenizer with proper message formatting."""

import time
from typing import List, Dict, Any, Union, Optional
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class AnthropicTokenizer:
    """Accurate Anthropic tokenizer with proper message formatting."""
    
    def __init__(self):
        self.encoder = None
        self.user_role_count = 0
        self.assistant_role_count = 0
        
        if TIKTOKEN_AVAILABLE:
            try:
                # Use tiktoken as approximation for Anthropic (they use similar tokenization)
                self.encoder = tiktoken.get_encoding("cl100k_base")
                # Approximate role token counts
                self.user_role_count = len(self.encoder.encode("\n\nHuman: "))
                self.assistant_role_count = len(self.encoder.encode("\n\nAssistant: "))
            except:
                pass
    
    def count_tokens(self, prompt: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Count tokens for Anthropic request with proper formatting."""
        start_time = time.time()
        
        if isinstance(prompt, str):
            result = self._count_text_tokens(prompt)
        else:
            result = self._count_chat_tokens(prompt)
        
        result["tokenization_duration_ms"] = (time.time() - start_time) * 1000
        return result
    
    def _count_text_tokens(self, text: str) -> Dict[str, Any]:
        """Count tokens for plain text."""
        if len(text) > 800000:
            raise ValueError("Content is too large to tokenize.")
        
        if not self.encoder:
            # Fallback: approximate token count (1 token ≈ 3.5 chars for Anthropic)
            return {
                "tokenizer": "character fallback",
                "token_count": int(len(text) / 3.5),
            }
        
        return {
            "tokenizer": "tiktoken (anthropic estimate)",
            "token_count": len(self.encoder.encode(text.normalize("NFKC"))),
        }
    
    def _count_chat_tokens(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Count tokens for chat messages with proper Anthropic formatting."""
        if not self.encoder:
            # Fallback for chat messages
            messages = request_data.get("messages", [])
            system = request_data.get("system", "")
            total_chars = len(system)
            
            for msg in messages:
                total_chars += len(str(msg.get("content", "")))
            
            return {
                "tokenizer": "character fallback",
                "token_count": int(total_chars / 3.5),
            }
        
        num_tokens = 0
        
        # Count system message tokens
        system = request_data.get("system", "")
        if system:
            num_tokens += len(self.encoder.encode(system.normalize("NFKC")))
        
        # Count message tokens
        messages = request_data.get("messages", [])
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Add role tokens
            if role == "user":
                num_tokens += self.user_role_count
            elif role == "assistant":
                num_tokens += self.assistant_role_count
            
            # Handle content (can be string or list)
            if isinstance(content, str):
                if len(content) > 800000 or num_tokens > 200000:
                    raise ValueError("Content is too large to tokenize.")
                num_tokens += len(self.encoder.encode(content.normalize("NFKC")))
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            if len(text) > 800000 or num_tokens > 200000:
                                raise ValueError("Content is too large to tokenize.")
                            num_tokens += len(self.encoder.encode(text.normalize("NFKC")))
                        elif item.get("type") == "image":
                            # Approximate image token cost (would need actual image processing)
                            num_tokens += 1000  # Base estimate for images
        
        # If last message is not assistant, add assistant priming
        if messages and messages[-1].get("role") != "assistant":
            num_tokens += self.assistant_role_count
        
        return {
            "tokenizer": "tiktoken (anthropic estimate)",
            "token_count": num_tokens,
        }
    
    def extract_text_from_request(self, request_data: Dict[str, Any]) -> str:
        """Extract text from Anthropic request for token counting."""
        text_parts = []
        
        # Add system message
        system = request_data.get("system", "")
        if system:
            text_parts.append(system)
        
        # Add messages
        messages = request_data.get("messages", [])
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