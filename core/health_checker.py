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
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.models_url = 'https://api.openai.com/v1/models'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check OpenAI API key health using /models endpoint"""
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
        """Classify OpenAI error types"""
        error_msg_lower = error_message.lower()
        
        # Authentication errors (permanent)
        if status_code in [401, 403]:
            return 'invalid_key', False
        
        # Rate limiting (temporary)
        if status_code == 429:
            return 'rate_limited', True
        
        # Quota/billing issues (permanent until resolved)
        if status_code == 402:
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
        
        # Other client errors (permanent)
        if status_code >= 400:
            return 'client_error', False
        
        return 'unknown_error', False


class AnthropicHealthChecker(BaseHealthChecker):
    """Health checker for Anthropic API"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        # Anthropic doesn't have a models endpoint, so we'll use a minimal message call
        self.messages_url = 'https://api.anthropic.com/v1/messages'
    
    def check_health(self, api_key: str) -> HealthResult:
        """Check Anthropic API key health using minimal message call"""
        start_time = time.time()
        
        # Minimal request to test auth without consuming many tokens
        test_payload = {
            "model": "claude-3-haiku-20240307",
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
        """Classify Anthropic error types"""
        error_msg_lower = error_message.lower()
        
        # Authentication errors (permanent)
        if status_code in [401, 403]:
            return 'invalid_key', False
        
        # Rate limiting (temporary)
        if status_code == 429:
            return 'rate_limited', True
        
        # Quota/billing issues (permanent until resolved)
        if status_code == 402:
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
        
        # Other client errors (permanent)
        if status_code >= 400:
            return 'client_error', False
        
        return 'unknown_error', False


class HealthCheckManager:
    """Manages health checking for multiple services"""
    
    def __init__(self):
        self.checkers = {
            'openai': OpenAIHealthChecker(),
            'anthropic': AnthropicHealthChecker()
        }
    
    def check_service_health(self, service: str, api_key: str) -> HealthResult:
        """Check health for a specific service and API key"""
        if service not in self.checkers:
            return HealthResult(
                status='unknown_service',
                error_message=f'No health checker available for service: {service}'
            )
        
        return self.checkers[service].check_health(api_key)
    
    def add_checker(self, service: str, checker: BaseHealthChecker):
        """Add a new health checker for a service"""
        self.checkers[service] = checker
    
    def get_supported_services(self) -> List[str]:
        """Get list of supported services"""
        return list(self.checkers.keys())


# Example usage for future services:
class DeepSeekHealthChecker(BaseHealthChecker):
    """Health checker for DeepSeek API"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.models_url = 'https://api.deepseek.com/v1/models'  # Example URL
    
    def check_health(self, api_key: str) -> HealthResult:
        # Similar implementation to OpenAI
        pass
    
    def classify_error(self, status_code: int, error_message: str) -> Tuple[str, bool]:
        # Service-specific error classification
        pass


# Global health check manager instance
health_manager = HealthCheckManager()