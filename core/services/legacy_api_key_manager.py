"""
Legacy API Key Manager Service for NyanProxy

This is the original APIKeyManager extracted from core/app.py
TODO: Migrate to use the sophisticated AIKeyManager from core/ai_key_manager.py
"""
import os
import time
import threading
from typing import Dict, List, Any
from datetime import datetime
from ..health_checker import health_manager


class LegacyAPIKeyManager:
    """Legacy API Key Manager - extracted from core/app.py"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.key_usage = {}  # Track usage per key for rate limiting
        self.failed_keys = set()  # Track temporarily failed keys
        self.key_health = {}  # Track detailed health status for each key
        self.lock = threading.Lock()  # Thread safety for key health updates
        self._initialize_key_health()
    
    def _parse_keys(self, key_string: str) -> List[str]:
        """Parse comma and line-separated keys"""
        if not key_string:
            return []
        
        keys = []
        # Split by both commas and newlines
        for line in key_string.replace(',', '\n').split('\n'):
            key = line.strip()
            if key and key not in keys:
                keys.append(key)
        return keys
    
    def _load_api_keys(self) -> Dict[str, List[str]]:
        """Load API keys from environment variables or config"""
        keys = {}
        
        # OpenAI keys - support comma/line separated format
        openai_keys = []
        
        # Try comma/line separated format first
        bulk_key = os.getenv('OPENAI_API_KEYS')
        if bulk_key:
            parsed_keys = self._parse_keys(bulk_key)
            openai_keys.extend(parsed_keys)
        
        # Fallback to single key
        if not openai_keys:
            single_key = os.getenv('OPENAI_API_KEY')
            if single_key:
                openai_keys.extend(self._parse_keys(single_key))
        
        # Only add services that have actual keys
        if openai_keys:
            keys['openai'] = list(set(openai_keys))  # Remove duplicates
        
        # Anthropic keys
        anthropic_keys = []
        bulk_key = os.getenv('ANTHROPIC_API_KEYS')
        if bulk_key:
            anthropic_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('ANTHROPIC_API_KEY')
        if single_key:
            anthropic_keys.extend(self._parse_keys(single_key))
        
        # Only add anthropic if it has keys
        if anthropic_keys:
            keys['anthropic'] = list(set(anthropic_keys))
        
        # Google keys
        google_keys = []
        bulk_key = os.getenv('GOOGLE_API_KEYS')
        if bulk_key:
            google_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('GOOGLE_API_KEY')
        if single_key:
            google_keys.extend(self._parse_keys(single_key))
        
        if google_keys:
            keys['google'] = list(set(google_keys))
        
        # Mistral keys
        mistral_keys = []
        bulk_key = os.getenv('MISTRAL_API_KEYS')
        if bulk_key:
            mistral_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('MISTRAL_API_KEY')
        if single_key:
            mistral_keys.extend(self._parse_keys(single_key))
        
        if mistral_keys:
            keys['mistral'] = list(set(mistral_keys))
        
        # Groq keys
        groq_keys = []
        bulk_key = os.getenv('GROQ_API_KEYS')
        if bulk_key:
            groq_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('GROQ_API_KEY')
        if single_key:
            groq_keys.extend(self._parse_keys(single_key))
        
        if groq_keys:
            keys['groq'] = list(set(groq_keys))
        
        # Cohere keys
        cohere_keys = []
        bulk_key = os.getenv('COHERE_API_KEYS')
        if bulk_key:
            cohere_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('COHERE_API_KEY')
        if single_key:
            cohere_keys.extend(self._parse_keys(single_key))
        
        if cohere_keys:
            keys['cohere'] = list(set(cohere_keys))
        
        return keys
    
    def _initialize_key_health(self):
        """Initialize health tracking for all keys"""
        with self.lock:
            for service, keys in self.api_keys.items():
                for key in keys:
                    if key and key not in self.key_health:
                        self.key_health[key] = {
                            'status': 'unknown',
                            'service': service,
                            'last_error': None,
                            'last_success': None,
                            'last_failure': None,
                            'failure_count': 0,
                            'consecutive_failures': 0,
                            'error_type': None
                        }
    
    def perform_proactive_health_check(self, service: str, api_key: str):
        """Perform proactive health check using the health checker system"""
        try:
            result = health_manager.check_service_health(service, api_key)
            
            # Update key health based on result
            if result.status == 'healthy':
                self.update_key_health(api_key, True)
            else:
                self.update_key_health(
                    api_key, 
                    False, 
                    result.status_code, 
                    result.error_message
                )
                
        except Exception as e:
            # If health check fails, mark as network error
            self.update_key_health(api_key, False, None, f"Health check failed: {str(e)}")
    
    def _classify_error(self, service: str, status_code: int, error_message: str) -> tuple:
        """Classify error type per specifications"""
        error_msg_lower = error_message.lower()
        
        if service == 'openai':
            # OpenAI: 401=Invalid, 429=Quota Exceeded
            if status_code == 401:
                return 'invalid_key', False  # Permanently remove
            elif status_code == 429:
                return 'quota_exceeded', False  # Permanently remove
                
        elif service == 'anthropic':
            # Anthropic: 401=Invalid, 429=Rate Limited, 403=Quota Exceeded
            if status_code == 401:
                return 'invalid_key', False  # Permanently remove
            elif status_code == 429:
                return 'rate_limited', True  # Keep in pool, temporary
            elif status_code == 403:
                return 'quota_exceeded', False  # Permanently remove
                
        elif service == 'google':
            # Google: 403=Invalid, 429=Rate Limited
            if status_code == 403:
                return 'invalid_key', False  # Permanently remove
            elif status_code == 429:
                return 'rate_limited', True  # Keep in pool, temporary
        
        # Fallback logic for other errors
        quota_indicators = ['quota exceeded', 'billing', 'insufficient_quota', 'usage limit']
        if any(indicator in error_msg_lower for indicator in quota_indicators):
            return 'quota_exceeded', False
        
        rate_limit_indicators = ['rate limit', 'too many requests', 'rate_limit_exceeded']
        if any(indicator in error_msg_lower for indicator in rate_limit_indicators):
            return 'rate_limited', True
        
        if status_code >= 500:
            return 'server_error', True
        
        if status_code >= 400:
            return 'client_error', False
        
        return 'unknown_error', False
    
    def get_api_key(self, service: str, exclude_failed: bool = True, exclude_key: str = None) -> str:
        """Get next available API key for the service with rate limit handling"""
        if service not in self.api_keys or not self.api_keys[service]:
            print(f"No API keys available for {service}")
            return None
        
        available_keys = [key for key in self.api_keys[service] if key]
        
        # Remove failed keys if requested
        if exclude_failed:
            available_keys = [key for key in available_keys if key not in self.failed_keys]
        
        # Exclude specific key (for retry with different key)
        if exclude_key:
            available_keys = [key for key in available_keys if key != exclude_key]
        
        if not available_keys:
            # If all keys failed, try again with failed keys (but still exclude the specific key)
            if exclude_failed:
                print(f"All {service} keys failed, trying with failed keys...")
                return self.get_api_key(service, exclude_failed=False, exclude_key=exclude_key)
            print(f"No available keys for {service}")
            return None
        
        # Simple round-robin selection
        if service not in self.key_usage:
            self.key_usage[service] = 0
        
        key_index = self.key_usage[service] % len(available_keys)
        selected_key = available_keys[key_index]
        self.key_usage[service] += 1
        
        print(f"Selected key {selected_key[:8]}...{selected_key[-4:]} for {service} ({key_index + 1}/{len(available_keys)} available)")
        return selected_key
    
    def update_key_health(self, key: str, success: bool, status_code: int = None, error_message: str = None):
        """Update key health status based on API response"""
        with self.lock:
            if key not in self.key_health:
                # Initialize if not exists
                service = next((svc for svc, keys in self.api_keys.items() if key in keys), 'unknown')
                self.key_health[key] = {
                    'status': 'unknown',
                    'service': service,
                    'last_error': None,
                    'last_success': None,
                    'last_failure': None,
                    'failure_count': 0,
                    'consecutive_failures': 0,
                    'error_type': None
                }
            
            health = self.key_health[key]
            now = datetime.now()
            
            if success:
                health['status'] = 'healthy'
                health['last_success'] = now
                health['consecutive_failures'] = 0
                health['error_type'] = None
                health['last_error'] = None
            else:
                health['last_failure'] = now
                health['failure_count'] += 1
                health['consecutive_failures'] += 1
                
                if status_code and error_message:
                    # We need service info for proper classification
                    error_type, is_retryable = self._classify_error('unknown', status_code, error_message)
                    health['error_type'] = error_type
                    health['status'] = error_type
                    health['last_error'] = error_message
                    
                    # Only mark as failed for retryable errors
                    if is_retryable:
                        self.failed_keys.add(key)
                        # Auto-recovery after some time
                        threading.Timer(300, lambda: self.failed_keys.discard(key)).start()
                else:
                    health['status'] = 'failed'
                    health['error_type'] = 'unknown_error'
    
    def mark_key_failed(self, service: str, key: str):
        """Mark a key as temporarily failed (legacy method)"""
        self.failed_keys.add(key)
        # Auto-recovery after some time (simplified)
        threading.Timer(300, lambda: self.failed_keys.discard(key)).start()
    
    def permanently_remove_key(self, service: str, key: str, reason: str):
        """Permanently remove a key from the available pool"""
        with self.lock:
            if service in self.api_keys and key in self.api_keys[service]:
                # Remove from active keys
                self.api_keys[service].remove(key)
                
                # Track removed key for stats
                if not hasattr(self, 'removed_keys'):
                    self.removed_keys = {}
                if service not in self.removed_keys:
                    self.removed_keys[service] = []
                
                self.removed_keys[service].append({
                    'key': key[:8] + "..." + key[-4:],
                    'reason': reason,
                    'removed_at': time.time()
                })
                
                # Also remove from failed keys set
                self.failed_keys.discard(key)
                
                print(f"PERMANENTLY REMOVED key {key[:8]}...{key[-4:]} from {service} pool (reason: {reason})")
                print(f"{service} pool now has {len(self.api_keys[service])} keys remaining")
    
    def handle_api_error(self, service: str, key: str, error_message: str, status_code: int = None) -> bool:
        """Handle API error and return True if should retry with different key"""
        print(f"API Error for {service}: {status_code} - {error_message}")
        
        # Update key health with failure
        self.update_key_health(key, False, status_code, error_message)
        
        # Check if error is retryable
        if status_code and error_message:
            error_type, is_retryable = self._classify_error(service, status_code, error_message)
            print(f"Error classified as: {error_type}, retryable: {is_retryable}")
            
            # Handle permanent failures - remove from key pool entirely
            if error_type in ['invalid_key', 'quota_exceeded']:
                self.permanently_remove_key(service, key, error_type)
                return True  # Still retryable with different key
            
            # Handle temporary failures - add to failed keys set
            elif error_type in ['rate_limited', 'server_error']:
                print(f"Temporarily marking {key[:8]}...{key[-4:]} as failed for {service}")
                self.failed_keys.add(key)
                return True
            
            return is_retryable
        
        # Legacy fallback for rate limiting
        rate_limit_indicators = [
            'rate limit', 'too many requests', 'quota exceeded',
            'rate_limit_exceeded', 'requests per minute', 'rpm', 'tpm'
        ]
        if any(indicator in error_message.lower() for indicator in rate_limit_indicators):
            self.mark_key_failed(service, key)
            return True
        
        return False
    
    def get_key_health_stats(self, service: str = None) -> Dict[str, Any]:
        """Get key health statistics for dashboard display"""
        with self.lock:
            if service:
                # Stats for specific service
                if service not in self.api_keys:
                    return {
                        'healthy': 0,
                        'rate_limited': 0,
                        'invalid_key': 0,
                        'quota_exceeded': 0,
                        'unknown': 0,
                        'avg_response_time': '0.00'
                    }
                
                stats = {
                    'healthy': 0,
                    'rate_limited': 0, 
                    'invalid_key': 0,
                    'quota_exceeded': 0,
                    'unknown': 0,
                    'total_response_time': 0.0,
                    'response_count': 0
                }
                
                # Count keys by health status
                for key in self.api_keys[service]:
                    if key in self.key_health:
                        health = self.key_health[key]
                        status = health.get('status', 'unknown')
                        
                        if status == 'healthy':
                            stats['healthy'] += 1
                        elif status in ['rate_limited', 'rate_limited_temporary']:
                            stats['rate_limited'] += 1
                        elif status in ['invalid_key', 'invalid']:
                            stats['invalid_key'] += 1
                        elif status in ['quota_exceeded', 'quota']:
                            stats['quota_exceeded'] += 1
                        else:
                            stats['unknown'] += 1
                        
                        # Accumulate response times for average
                        if health.get('last_response_time'):
                            stats['total_response_time'] += health['last_response_time']
                            stats['response_count'] += 1
                    else:
                        # Key not health checked yet
                        stats['healthy'] += 1
                
                # Add removed keys to the appropriate categories
                if hasattr(self, 'removed_keys') and service in self.removed_keys:
                    for removed_key in self.removed_keys[service]:
                        reason = removed_key['reason']
                        if reason == 'invalid_key':
                            stats['invalid_key'] += 1
                        elif reason == 'quota_exceeded':
                            stats['quota_exceeded'] += 1
                
                # Calculate average response time
                if stats['response_count'] > 0:
                    avg_response_time = stats['total_response_time'] / stats['response_count']
                    stats['avg_response_time'] = f"{avg_response_time:.2f}"
                else:
                    stats['avg_response_time'] = '0.00'
                
                # Remove internal calculation fields
                del stats['total_response_time']
                del stats['response_count']
                
                return stats
            
            else:
                # Stats for all services
                all_stats = {}
                for svc in self.api_keys.keys():
                    all_stats[svc] = self.get_key_health_stats(svc)
                return all_stats