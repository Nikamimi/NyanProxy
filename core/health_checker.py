"""
Health checking system for various AI service providers
"""
import requests
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class HealthResult:
    """Result of a health check"""
    status: str  # 'healthy', 'invalid_key', 'rate_limited', 'quota_exceeded', 'network_error'
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    is_retryable: bool = False


class BaseHealthChecker(ABC):
    """Base class for service-specific health checkers"""
    
    @abstractmethod
    def check_health(self, api_key: str) -> HealthResult:
        """Check health of a specific API key"""
        pass
    
    @abstractmethod
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify error type and return (status, is_retryable)"""
        pass


class OpenAIHealthChecker(BaseHealthChecker):
    """Health checker for OpenAI API"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.completions_url = 'https://api.openai.com/v1/chat/completions'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check OpenAI API key health using minimal completion call"""
        start_time = time.time()
        
        # Minimal completion request to test API functionality
        test_payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                self.completions_url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json=test_payload,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return HealthResult(
                    status='healthy',
                    response_time=response_time,
                    status_code=200
                )
            else:
                status, is_retryable = self.classify_error(response.status_code, response.text)
                return HealthResult(
                    status=status,
                    error_message=response.text,
                    response_time=response_time,
                    status_code=response.status_code,
                    is_retryable=is_retryable
                )
                
        except requests.exceptions.Timeout:
            return HealthResult(
                status='network_error',
                error_message='Request timeout',
                response_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return HealthResult(
                status='network_error',
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify OpenAI error types according to requirements"""
        error_msg_lower = error_message.lower()
        
        # 401 = Invalid (permanent)
        if status_code == 401:
            return 'invalid_key', False
        
        # 429 = Quota Exceeded (permanent)
        if status_code == 429:
            return 'quota_exceeded', False
        
        # Check for quota-related messages
        quota_indicators = [
            'exceeded your current quota',
            'insufficient_quota',
            'billing_hard_limit_reached',
            'quota exceeded',
            'usage limit'
        ]
        if any(indicator in error_msg_lower for indicator in quota_indicators):
            return 'quota_exceeded', False
        
        # Server errors (potentially retryable)
        if status_code >= 500:
            return 'server_error', True
        
        # Other client errors (keep in pool)
        if status_code >= 400:
            return 'client_error', True
        
        return 'unknown_error', True


class AnthropicHealthChecker(BaseHealthChecker):
    """Health checker for Anthropic API"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        # Anthropic doesn't have a models endpoint, so we'll use a minimal message call
        self.messages_url = 'https://api.anthropic.com/v1/messages'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check Anthropic API key health using minimal message call"""
        start_time = time.time()
        
        # Minimal request to test auth without consuming many tokens
        test_payload = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "Hi"}]
        }
        
        try:
            response = requests.post(
                self.messages_url,
                headers={
                    'x-api-key': api_key,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json=test_payload,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return HealthResult(
                    status='healthy',
                    response_time=response_time,
                    status_code=200
                )
            else:
                status, is_retryable = self.classify_error(response.status_code, response.text)
                return HealthResult(
                    status=status,
                    error_message=response.text,
                    response_time=response_time,
                    status_code=response.status_code,
                    is_retryable=is_retryable
                )
                
        except requests.exceptions.Timeout:
            return HealthResult(
                status='network_error',
                error_message='Request timeout',
                response_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return HealthResult(
                status='network_error',
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify Anthropic error types according to requirements"""
        error_msg_lower = error_message.lower()
        
        # 401 = Invalid (permanent)
        if status_code == 401:
            return 'invalid_key', False
        
        # 429 = Rate-Limited (keep in pool, temporary)
        if status_code == 429:
            return 'rate_limited', True
        
        # 403 = Quota Exceeded (permanent)
        if status_code == 403:
            return 'quota_exceeded', False
        
        # Check for quota-related messages
        quota_indicators = [
            'credit_balance_too_low',
            'insufficient_credit',
            'quota exceeded',
            'usage limit',
            'billing'
        ]
        if any(indicator in error_msg_lower for indicator in quota_indicators):
            return 'quota_exceeded', False
        
        # Server errors (potentially retryable)
        if status_code >= 500:
            return 'server_error', True
        
        # Other client errors (keep in pool)
        if status_code >= 400:
            return 'client_error', True
        
        return 'unknown_error', True


class HealthCheckManager:
    """Manages health checking for multiple services"""
    
    def __init__(self):
        self.checkers = {
            'openai': OpenAIHealthChecker(),
            'anthropic': AnthropicHealthChecker(),
            'google': GoogleHealthChecker(),
            'mistral': MistralHealthChecker(),
            'groq': GroqHealthChecker(),
            'cohere': CohereHealthChecker()
        }
    
    def _quick_validate_key_format(self, service: str, api_key: str) -> Optional[HealthResult]:
        """Quick format validation to fail fast on obviously invalid keys"""
        if not api_key or len(api_key.strip()) < 10:
            return HealthResult(
                status='invalid_key',
                error_message='API key too short or empty',
                response_time=0.001
            )
        
        # Service-specific format checks
        if service == 'openai' and not api_key.startswith('sk-'):
            return HealthResult(
                status='invalid_key', 
                error_message='OpenAI keys must start with sk-',
                response_time=0.001
            )
        elif service == 'anthropic' and not api_key.startswith('sk-ant-'):
            return HealthResult(
                status='invalid_key',
                error_message='Anthropic keys must start with sk-ant-',
                response_time=0.001
            )
        elif service == 'google' and not (api_key.startswith('AIzaSy') and len(api_key) == 39):
            return HealthResult(
                status='invalid_key',
                error_message='Google API keys must start with AIzaSy and be 39 characters',
                response_time=0.001
            )
        
        return None  # Passed basic validation
    
    def check_service_health(self, service: str, api_key: str) -> HealthResult:
        """Check health for a specific service and API key"""
        if service not in self.checkers:
            return HealthResult(
                status='unknown_service',
                error_message=f'No health checker available for service: {service}'
            )
        
        # Quick format validation first (fail fast)
        quick_result = self._quick_validate_key_format(service, api_key)
        if quick_result:
            return quick_result
        
        # Full health check if format looks valid
        return self.checkers[service].check_health(api_key)
    
    def add_checker(self, service: str, checker: BaseHealthChecker):
        """Add a new health checker for a service"""
        self.checkers[service] = checker
    
    def get_supported_services(self) -> List[str]:
        """Get list of supported services"""
        return list(self.checkers.keys())


class GoogleHealthChecker(BaseHealthChecker):
    """Health checker for Google Gemini API"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.models_url = 'https://generativelanguage.googleapis.com/v1/models'
        self.generate_url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check Google API key health using fast models endpoint first, fallback to generation"""
        start_time = time.time()
        
        try:
            # Try fast models endpoint first (much faster than generation)
            response = requests.get(
                f'{self.models_url}?key={api_key}',
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return HealthResult(
                    status='healthy',
                    response_time=response_time,
                    status_code=200
                )
            else:
                status, is_retryable = self.classify_error(response.status_code, response.text)
                return HealthResult(
                    status=status,
                    error_message=response.text,
                    response_time=response_time,
                    status_code=response.status_code,
                    is_retryable=is_retryable
                )
                
        except requests.exceptions.Timeout:
            return HealthResult(
                status='network_error',
                error_message='Request timeout',
                response_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return HealthResult(
                status='network_error',
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify Google error types according to requirements"""
        error_msg_lower = error_message.lower()
        
        # 403 = Invalid (permanent)
        if status_code == 403:
            return 'invalid_key', False
        
        # 429 = Rate-Limited (keep in pool, temporary)
        if status_code == 429:
            return 'rate_limited', True
        
        # No Quota Exceeded for Google according to requirements
        
        # Server errors (potentially retryable)
        if status_code >= 500:
            return 'server_error', True
        
        # Other client errors (keep in pool)
        if status_code >= 400:
            return 'client_error', True
        
        return 'unknown_error', True


class MistralHealthChecker(BaseHealthChecker):
    """Health checker for Mistral API"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.models_url = 'https://api.mistral.ai/v1/models'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check Mistral API key health using /models endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(
                self.models_url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return HealthResult(
                    status='healthy',
                    response_time=response_time,
                    status_code=200
                )
            else:
                status, is_retryable = self.classify_error(response.status_code, response.text)
                return HealthResult(
                    status=status,
                    error_message=response.text,
                    response_time=response_time,
                    status_code=response.status_code,
                    is_retryable=is_retryable
                )
                
        except requests.exceptions.Timeout:
            return HealthResult(
                status='network_error',
                error_message='Request timeout',
                response_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return HealthResult(
                status='network_error',
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify Mistral error types"""
        error_msg_lower = error_message.lower()
        
        if status_code in [401, 403]:
            return 'invalid_key', False
        if status_code == 429:
            return 'rate_limited', True
        if status_code == 402:
            return 'quota_exceeded', False
        if 'quota exceeded' in error_msg_lower or 'insufficient balance' in error_msg_lower:
            return 'quota_exceeded', False
        if status_code >= 500:
            return 'server_error', True
        if status_code >= 400:
            return 'client_error', False
        return 'unknown_error', False


class GroqHealthChecker(BaseHealthChecker):
    """Health checker for Groq API"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.models_url = 'https://api.groq.com/openai/v1/models'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check Groq API key health using /models endpoint"""
        start_time = time.time()
        
        try:
            response = requests.get(
                self.models_url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return HealthResult(
                    status='healthy',
                    response_time=response_time,
                    status_code=200
                )
            else:
                status, is_retryable = self.classify_error(response.status_code, response.text)
                return HealthResult(
                    status=status,
                    error_message=response.text,
                    response_time=response_time,
                    status_code=response.status_code,
                    is_retryable=is_retryable
                )
                
        except requests.exceptions.Timeout:
            return HealthResult(
                status='network_error',
                error_message='Request timeout',
                response_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return HealthResult(
                status='network_error',
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify Groq error types"""
        error_msg_lower = error_message.lower()
        
        if status_code in [401, 403]:
            return 'invalid_key', False
        if status_code == 429:
            return 'rate_limited', True
        if status_code == 402:
            return 'quota_exceeded', False
        if 'quota exceeded' in error_msg_lower or 'rate limit' in error_msg_lower:
            return 'quota_exceeded', False
        if status_code >= 500:
            return 'server_error', True
        if status_code >= 400:
            return 'client_error', False
        return 'unknown_error', False


class CohereHealthChecker(BaseHealthChecker):
    """Health checker for Cohere API"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        # Cohere doesn't have a models endpoint, use a minimal generation call
        self.generate_url = 'https://api.cohere.ai/v1/generate'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check Cohere API key health using minimal generate call"""
        start_time = time.time()
        
        # Minimal request to test auth without consuming many tokens
        test_payload = {
            "model": "command",
            "prompt": "Hi",
            "max_tokens": 1,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(
                self.generate_url,
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json=test_payload,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return HealthResult(
                    status='healthy',
                    response_time=response_time,
                    status_code=200
                )
            else:
                status, is_retryable = self.classify_error(response.status_code, response.text)
                return HealthResult(
                    status=status,
                    error_message=response.text,
                    response_time=response_time,
                    status_code=response.status_code,
                    is_retryable=is_retryable
                )
                
        except requests.exceptions.Timeout:
            return HealthResult(
                status='network_error',
                error_message='Request timeout',
                response_time=time.time() - start_time
            )
        except requests.exceptions.RequestException as e:
            return HealthResult(
                status='network_error',
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        """Classify Cohere error types"""
        error_msg_lower = error_message.lower()
        
        if status_code in [401, 403]:
            return 'invalid_key', False
        if status_code == 429:
            return 'rate_limited', True
        if status_code == 402:
            return 'quota_exceeded', False
        if 'quota exceeded' in error_msg_lower or 'insufficient credits' in error_msg_lower:
            return 'quota_exceeded', False
        if status_code >= 500:
            return 'server_error', True
        if status_code >= 400:
            return 'client_error', False
        return 'unknown_error', False


# Global health check manager instance
health_manager = HealthCheckManager()