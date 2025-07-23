"""
Health Route Handler for NyanProxy

Handles all health check and monitoring endpoints
Extracted from core/app.py for better modularity
"""
import time
from datetime import datetime
from typing import Dict, Any, Tuple
from flask import jsonify

from .base_handler import BaseHandler


class HealthHandler(BaseHandler):
    """Handler for health check and monitoring routes"""
    
    def basic_health(self) -> Tuple[Any, int]:
        """Basic health check endpoint"""
        start_time = self.track_request_start()
        
        try:
            result = {"status": "healthy", "service": "AI Proxy"}
            self.track_request_end('health', start_time)
            return self.create_success_response(result), 200
            
        except Exception as e:
            self.track_request_end('health', start_time, error=True)
            return self.create_error_response(str(e), 500)
    
    def key_status(self) -> Tuple[Any, int]:
        """Check which API keys are configured"""
        start_time = self.track_request_start()
        
        try:
            status = {}
            for service, keys in self.key_manager.api_keys.items():
                status[service] = {
                    'configured': len([k for k in keys if k]) > 0,
                    'count': len([k for k in keys if k])
                }
            
            self.track_request_end('key_status', start_time)
            return self.create_success_response(status), 200
            
        except Exception as e:
            self.track_request_end('key_status', start_time, error=True)
            return self.create_error_response(str(e), 500)
    
    def key_health(self) -> Tuple[Any, int]:
        """Get detailed health status for all keys"""
        start_time = self.track_request_start()
        
        try:
            # Get health data with safe serialization
            health_data = {}
            with self.key_manager.lock:
                for key, health in self.key_manager.key_health.items():
                    # Mask the key for security (show only first 8 chars)
                    masked_key = key[:8] + '...' if len(key) > 8 else key
                    health_data[masked_key] = {
                        'status': health['status'],
                        'service': health['service'],
                        'last_success': health['last_success'].isoformat() if health['last_success'] else None,
                        'last_failure': health['last_failure'].isoformat() if health['last_failure'] else None,
                        'failure_count': health['failure_count'],
                        'consecutive_failures': health['consecutive_failures'],  
                        'error_type': health['error_type'],
                        'last_error': health['last_error'][:100] if health['last_error'] else None  # Truncate error message
                    }
            
            self.track_request_end('key_health', start_time)
            return self.create_success_response(health_data), 200
            
        except Exception as e:
            self.track_request_end('key_health', start_time, error=True)
            return self.create_error_response(str(e), 500)
    
    def debug_keys(self) -> Tuple[Any, int]:
        """Debug endpoint to check key loading"""
        start_time = self.track_request_start()
        
        try:
            debug_info = {}
            for service, keys in self.key_manager.api_keys.items():
                debug_info[service] = {
                    'total_keys': len(keys),
                    'valid_keys': len([k for k in keys if k]),
                    'first_few_chars': [k[:8] + '...' if len(k) > 8 else k for k in keys[:3] if k]  # Show first 3 for debugging
                }
            
            self.track_request_end('debug_keys', start_time)
            return self.create_success_response(debug_info), 200
            
        except Exception as e:
            self.track_request_end('debug_keys', start_time, error=True)
            return self.create_error_response(str(e), 500)