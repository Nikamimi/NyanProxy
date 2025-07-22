"""
Integrated API Manager that replaces the existing key management system
with the new comprehensive retry logic and key pool management
"""
import os
import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from .ai_key_manager import get_ai_key_manager, AIKeyManager
from .retry_logic import UniversalRetrySystem, RetryResult


class IntegratedAPIManager:
    """
    Drop-in replacement for the existing APIKeyManager with enhanced features
    """
    
    def __init__(self, firebase_db=None):
        """Initialize with the new AI Key Management System"""
        print("🔄 Initializing Integrated API Manager...")
        
        # Initialize the new AI Key Management System
        self.ai_key_manager = get_ai_key_manager(firebase_db)
        
        # Load API keys from environment and initialize them
        self._initialize_from_environment()
        
        # Legacy compatibility attributes
        self.api_keys = {}
        self.key_usage = {}
        self.failed_keys = set()
        self.key_health = {}
        self.lock = threading.Lock()
        
        print("✅ Integrated API Manager ready")
    
    def _parse_keys(self, key_string: str) -> List[str]:
        """Parse comma and line-separated keys (legacy compatibility)"""
        if not key_string:
            return []
        
        keys = []
        for line in key_string.replace(',', '\n').split('\n'):
            key = line.strip()
            if key and key not in keys:
                keys.append(key)
        return keys
    
    def _initialize_from_environment(self):
        """Initialize API keys from environment variables"""
        print("🔧 Loading API keys from environment...")
        
        # OpenAI keys
        openai_keys = []
        bulk_key = os.getenv('OPENAI_API_KEYS')
        if bulk_key:
            openai_keys.extend(self._parse_keys(bulk_key))
        
        # Fallback to single key
        if not openai_keys:
            single_key = os.getenv('OPENAI_API_KEY')
            if single_key:
                openai_keys.extend(self._parse_keys(single_key))
        
        if openai_keys:
            result = self.ai_key_manager.initialize_provider_keys('openai', openai_keys)
            print(f"✅ OpenAI: {result['healthy_keys_added']}/{result['total_keys_provided']} keys ready")
        
        # Anthropic keys
        anthropic_keys = []
        bulk_key = os.getenv('ANTHROPIC_API_KEYS')
        if bulk_key:
            anthropic_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('ANTHROPIC_API_KEY')
        if single_key and not anthropic_keys:
            anthropic_keys.extend(self._parse_keys(single_key))
        
        if anthropic_keys:
            result = self.ai_key_manager.initialize_provider_keys('anthropic', anthropic_keys)
            print(f"✅ Anthropic: {result['healthy_keys_added']}/{result['total_keys_provided']} keys ready")
        
        # Google keys
        google_keys = []
        bulk_key = os.getenv('GOOGLE_API_KEYS')
        if bulk_key:
            google_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('GOOGLE_API_KEY')
        if single_key and not google_keys:
            google_keys.extend(self._parse_keys(single_key))
        
        if google_keys:
            result = self.ai_key_manager.initialize_provider_keys('google', google_keys)
            print(f"✅ Google: {result['healthy_keys_added']}/{result['total_keys_provided']} keys ready")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get a healthy API key for the service (legacy compatibility)"""
        # Use the new key pool manager to get a healthy key
        key_obj = self.ai_key_manager.key_pool.get_healthy_key(service)
        return key_obj.key if key_obj else None
    
    def handle_api_error(self, service: str, key: str, error_message: str, status_code: int = None) -> bool:
        """
        Handle API error with the new system
        Returns True if should retry with different key
        """
        print(f"🔴 API Error for {service}: {status_code} - {error_message}")
        
        # Classify the error using the new system
        retry_system = self.ai_key_manager.retry_system
        error_status, should_retry = retry_system._classify_error(service, 
            type('MockException', (), {
                'status_code': status_code,
                '__str__': lambda: error_message
            })()
        )
        
        # Mark the key result in the new system
        self.ai_key_manager.key_pool.mark_key_result(
            provider=service,
            key=key,
            success=False,
            error_status=error_status,
            error_message=error_message
        )
        
        print(f"🔍 Error classified as: {error_status}, should_retry: {should_retry}")
        return should_retry
    
    def update_key_health(self, key: str, success: bool, status_code: int = None, error_message: str = None):
        """Update key health (legacy compatibility)"""
        # The new system handles this automatically through mark_key_result
        pass
    
    def mark_key_failed(self, service: str, key: str):
        """Mark key as failed (legacy compatibility)"""
        # The new system handles this automatically
        pass
    
    def execute_api_call_with_retry(self, service: str, api_call_func, *args, **kwargs) -> RetryResult:
        """
        Execute an API call with the new retry system
        This is the main method that should be used for all API calls
        """
        return self.ai_key_manager.execute_api_call(service, api_call_func, *args, **kwargs)
    
    def get_service_stats(self, service: str = None) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        if service:
            return self.ai_key_manager.get_provider_analytics(service)
        else:
            return self.ai_key_manager.get_comprehensive_status()
    
    def force_health_check(self):
        """Force immediate health check on all keys"""
        return self.ai_key_manager.force_health_check()


# Global instance for backward compatibility
integrated_manager = None

def get_integrated_manager(firebase_db=None) -> IntegratedAPIManager:
    """Get or create the global integrated manager"""
    global integrated_manager
    if integrated_manager is None:
        integrated_manager = IntegratedAPIManager(firebase_db)
    return integrated_manager


# API call wrapper functions for easy integration
def make_anthropic_call(request_json: Dict, headers: Dict = None) -> RetryResult:
    """
    Make an Anthropic API call with full retry logic
    This replaces the manual retry loop in the original code
    """
    def anthropic_api_call(request_json: Dict, headers: Dict, api_key: str):
        """Actual Anthropic API call"""
        import requests
        
        # Set up headers with the API key
        call_headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        if headers:
            call_headers.update(headers)
        
        # Make the API call
        session = requests.Session()
        response = session.post(
            'https://api.anthropic.com/v1/messages',
            headers=call_headers,
            json=request_json,
            stream=request_json.get('stream', False),
            timeout=(10, 30)
        )
        
        # Check for errors
        if response.status_code >= 400:
            error = Exception(response.text)
            error.status_code = response.status_code
            raise error
        
        # Return successful response
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.content,
            'json': response.json() if response.headers.get('content-type', '').startswith('application/json') else None,
            'text': response.text
        }
    
    # Execute with retry logic
    manager = get_integrated_manager()
    return manager.execute_api_call_with_retry('anthropic', anthropic_api_call, 
                                              request_json=request_json, 
                                              headers=headers)


def make_openai_call(request_json: Dict, headers: Dict = None) -> RetryResult:
    """
    Make an OpenAI API call with full retry logic
    """
    def openai_api_call(request_json: Dict, headers: Dict, api_key: str):
        """Actual OpenAI API call"""
        import requests
        
        # Set up headers with the API key
        call_headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        if headers:
            call_headers.update(headers)
        
        # Make the API call
        session = requests.Session()
        response = session.post(
            'https://api.openai.com/v1/chat/completions',
            headers=call_headers,
            json=request_json,
            stream=request_json.get('stream', False),
            timeout=(10, 30)
        )
        
        # Check for errors
        if response.status_code >= 400:
            error = Exception(response.text)
            error.status_code = response.status_code
            raise error
        
        # Return successful response
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.content,
            'json': response.json() if response.headers.get('content-type', '').startswith('application/json') else None,
            'text': response.text
        }
    
    # Execute with retry logic
    manager = get_integrated_manager()
    return manager.execute_api_call_with_retry('openai', openai_api_call,
                                              request_json=request_json,
                                              headers=headers)


def make_google_call(request_json: Dict, model_path: str, api_key_param: str = None) -> RetryResult:
    """
    Make a Google API call with full retry logic
    """
    def google_api_call(request_json: Dict, model_path: str, api_key_param: str, api_key: str):
        """Actual Google API call"""
        import requests
        
        # Construct URL with API key
        url = f'https://generativelanguage.googleapis.com/v1/{model_path}'
        params = {'key': api_key}
        
        # Make the API call
        session = requests.Session()
        response = session.post(
            url,
            params=params,
            json=request_json,
            timeout=(10, 30)
        )
        
        # Check for errors
        if response.status_code >= 400:
            error = Exception(response.text)
            error.status_code = response.status_code
            raise error
        
        # Return successful response
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.content,
            'json': response.json() if response.headers.get('content-type', '').startswith('application/json') else None,
            'text': response.text
        }
    
    # Execute with retry logic
    manager = get_integrated_manager()
    return manager.execute_api_call_with_retry('google', google_api_call,
                                              request_json=request_json,
                                              model_path=model_path,
                                              api_key_param=api_key_param)