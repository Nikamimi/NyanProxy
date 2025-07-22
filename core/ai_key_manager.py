"""
Complete AI Key Management System Integration
Provides a unified interface for all key management, health checking, and retry logic
"""
import os
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from .key_pool_manager import KeyPoolManager
from .retry_logic import UniversalRetrySystem, RetryResult
from .health_checker import HealthCheckManager


class AIKeyManager:
    """
    Complete AI Key Management System
    
    This class provides a unified interface for:
    1. Initial key health checks
    2. Key pool management with real-time updates
    3. Universal retry logic
    4. Health monitoring and reporting
    """
    
    def __init__(self, firebase_db=None):
        """Initialize the AI Key Manager"""
        self.firebase_db = firebase_db
        
        # Initialize core components
        self.key_pool = KeyPoolManager(firebase_db)
        self.retry_system = UniversalRetrySystem(self.key_pool)
        self.health_manager = HealthCheckManager()
        
        # Track initialization status
        self.initialized_providers = set()
        
        print("🚀 AI Key Manager initialized")
        print(f"   - Max retries: {self.retry_system.max_retries}")
        print(f"   - Health check interval: {self.key_pool.health_check_interval} minutes")
        print(f"   - Firebase available: {FIREBASE_AVAILABLE and firebase_db is not None}")
    
    def initialize_provider_keys(self, provider: str, api_keys: List[str]) -> Dict[str, Any]:
        """
        Initialize API keys for a provider with health checks
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'google')
            api_keys: List of API keys to initialize
            
        Returns:
            Dictionary with initialization results
        """
        print(f"🔧 Initializing {len(api_keys)} keys for {provider}")
        
        # Step 1: Perform initial health checks
        healthy_keys = []
        failed_keys = []
        
        for i, key in enumerate(api_keys):
            print(f"🏥 Health checking key {i+1}/{len(api_keys)} for {provider}...")
            
            try:
                health_result = self.health_manager.check_service_health(provider, key)
                
                if health_result.status == 'healthy':
                    healthy_keys.append(key)
                    print(f"   ✅ Key {key[:8]}... is healthy ({health_result.response_time:.2f}s)")
                else:
                    failed_keys.append({
                        'key': key[:8] + "...",
                        'status': health_result.status,
                        'error': health_result.error_message
                    })
                    print(f"   ❌ Key {key[:8]}... failed: {health_result.status}")
                    
            except Exception as e:
                failed_keys.append({
                    'key': key[:8] + "...",
                    'status': 'error',
                    'error': str(e)
                })
                print(f"   ❌ Key {key[:8]}... error: {e}")
        
        # Step 2: Add healthy keys to pool
        if healthy_keys:
            added_count = self.key_pool.add_keys(provider, healthy_keys)
            print(f"✅ Added {added_count} healthy keys to {provider} pool")
        else:
            print(f"⚠️ No healthy keys found for {provider}")
        
        # Mark provider as initialized
        self.initialized_providers.add(provider)
        
        # Step 3: Return summary
        result = {
            'provider': provider,
            'total_keys_provided': len(api_keys),
            'healthy_keys_added': len(healthy_keys),
            'failed_keys': len(failed_keys),
            'failed_key_details': failed_keys,
            'pool_status': self.key_pool.get_pool_status(provider)
        }
        
        print(f"📊 {provider} initialization complete:")
        print(f"   - Healthy: {len(healthy_keys)}/{len(api_keys)}")
        print(f"   - Failed: {len(failed_keys)}/{len(api_keys)}")
        
        return result
    
    def execute_api_call(self, provider: str, api_call_func: Callable, *args, **kwargs) -> RetryResult:
        """
        Execute an API call with full retry logic and key management
        
        Args:
            provider: AI provider name
            api_call_func: Function that makes the API call
            *args, **kwargs: Arguments for the API call
            
        Returns:
            RetryResult with success status and response data
        """
        if provider not in self.initialized_providers:
            return RetryResult(
                success=False,
                error_message=f"Provider {provider} not initialized. Call initialize_provider_keys() first.",
                attempts_made=0
            )
        
        return self.retry_system.execute_with_retry(provider, api_call_func, *args, **kwargs)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all key pools and systems"""
        return {
            'system_info': {
                'max_retries': self.retry_system.max_retries,
                'initialized_providers': list(self.initialized_providers),
                'firebase_available': FIREBASE_AVAILABLE and self.firebase_db is not None,
                'health_check_interval_minutes': self.key_pool.health_check_interval
            },
            'key_pools': self.key_pool.get_pool_status(),
            'removed_keys': self.key_pool.get_removed_keys(),
            'global_stats': self.key_pool.stats
        }
    
    def force_health_check(self, provider: str = None) -> Dict[str, Any]:
        """Force immediate health check on all or specific provider keys"""
        print("🏥 Forcing immediate health check...")
        
        if provider:
            # Check specific provider
            if provider not in self.key_pool.key_pools:
                return {'error': f'Provider {provider} not found'}
            
            print(f"🏥 Checking {provider} keys...")
            # We'll need to implement provider-specific health checks in key_pool_manager
            # For now, trigger the general health check
            self.key_pool.perform_health_checks()
            return self.key_pool.get_pool_status(provider)
        else:
            # Check all providers
            self.key_pool.perform_health_checks()
            return self.key_pool.get_pool_status()
    
    def add_new_keys(self, provider: str, new_keys: List[str]) -> Dict[str, Any]:
        """Add new keys to an existing provider pool with health checks"""
        print(f"🔑 Adding {len(new_keys)} new keys to {provider}")
        
        # Health check new keys before adding
        healthy_keys = []
        failed_keys = []
        
        for key in new_keys:
            try:
                health_result = self.health_manager.check_service_health(provider, key)
                
                if health_result.status == 'healthy':
                    healthy_keys.append(key)
                    print(f"   ✅ New key {key[:8]}... is healthy")
                else:
                    failed_keys.append({
                        'key': key[:8] + "...",
                        'status': health_result.status,
                        'error': health_result.error_message
                    })
                    print(f"   ❌ New key {key[:8]}... failed: {health_result.status}")
                    
            except Exception as e:
                failed_keys.append({
                    'key': key[:8] + "...",
                    'status': 'error',
                    'error': str(e)
                })
                print(f"   ❌ New key {key[:8]}... error: {e}")
        
        # Add healthy keys to pool
        added_count = 0
        if healthy_keys:
            added_count = self.key_pool.add_keys(provider, healthy_keys)
        
        return {
            'provider': provider,
            'new_keys_provided': len(new_keys),
            'healthy_keys_added': added_count,
            'failed_keys': len(failed_keys),
            'failed_key_details': failed_keys,
            'updated_pool_status': self.key_pool.get_pool_status(provider)
        }
    
    def remove_key(self, provider: str, key_identifier: str) -> bool:
        """Manually remove a key from the pool"""
        return self.key_pool.force_remove_key(provider, key_identifier)
    
    def update_configuration(self, **config) -> Dict[str, Any]:
        """Update system configuration"""
        updated = {}
        
        if 'max_retries' in config:
            old_retries = self.retry_system.max_retries
            self.retry_system.set_max_retries(config['max_retries'])
            updated['max_retries'] = {
                'old': old_retries,
                'new': self.retry_system.max_retries
            }
        
        if 'health_check_interval' in config:
            old_interval = self.key_pool.health_check_interval
            self.key_pool.health_check_interval = max(1, config['health_check_interval'])
            updated['health_check_interval'] = {
                'old': old_interval,
                'new': self.key_pool.health_check_interval
            }
        
        return {
            'updated_settings': updated,
            'current_config': {
                'max_retries': self.retry_system.max_retries,
                'health_check_interval': self.key_pool.health_check_interval
            }
        }
    
    def get_provider_analytics(self, provider: str, days: int = 7) -> Dict[str, Any]:
        """Get analytics for a specific provider"""
        pool_status = self.key_pool.get_pool_status(provider)
        if 'error' in pool_status:
            return pool_status
        
        # Calculate analytics
        total_keys = pool_status['total_keys']
        healthy_keys = pool_status['healthy_keys']
        removed_keys = pool_status['removed_keys']
        
        health_percentage = (healthy_keys / total_keys * 100) if total_keys > 0 else 0
        
        # Key performance metrics
        key_performance = []
        for key_info in pool_status['keys']:
            key_performance.append({
                'masked_key': key_info['masked_key'],
                'success_rate': key_info['success_rate'],
                'total_requests': key_info['success_count'] + key_info['error_count'],
                'status': key_info['status'],
                'avg_response_time': key_info['response_time']
            })
        
        # Sort by success rate descending
        key_performance.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return {
            'provider': provider,
            'summary': {
                'total_keys': total_keys,
                'healthy_keys': healthy_keys,
                'removed_keys': removed_keys,
                'health_percentage': round(health_percentage, 1)
            },
            'key_performance': key_performance,
            'status_breakdown': pool_status['status_breakdown']
        }


# Global instance for easy importing
_global_ai_key_manager = None

def get_ai_key_manager(firebase_db=None) -> AIKeyManager:
    """Get or create the global AI Key Manager instance"""
    global _global_ai_key_manager
    
    if _global_ai_key_manager is None:
        _global_ai_key_manager = AIKeyManager(firebase_db)
    
    return _global_ai_key_manager


# Convenience functions for direct usage
def initialize_openai_keys(api_keys: List[str]) -> Dict[str, Any]:
    """Initialize OpenAI keys"""
    manager = get_ai_key_manager()
    return manager.initialize_provider_keys('openai', api_keys)


def initialize_anthropic_keys(api_keys: List[str]) -> Dict[str, Any]:
    """Initialize Anthropic keys"""
    manager = get_ai_key_manager()
    return manager.initialize_provider_keys('anthropic', api_keys)


def initialize_google_keys(api_keys: List[str]) -> Dict[str, Any]:
    """Initialize Google keys"""
    manager = get_ai_key_manager()
    return manager.initialize_provider_keys('google', api_keys)


def execute_openai_call(api_call_func: Callable, *args, **kwargs) -> RetryResult:
    """Execute OpenAI API call with retry logic"""
    manager = get_ai_key_manager()
    return manager.execute_api_call('openai', api_call_func, *args, **kwargs)


def execute_anthropic_call(api_call_func: Callable, *args, **kwargs) -> RetryResult:
    """Execute Anthropic API call with retry logic"""
    manager = get_ai_key_manager()
    return manager.execute_api_call('anthropic', api_call_func, *args, **kwargs)


def execute_google_call(api_call_func: Callable, *args, **kwargs) -> RetryResult:
    """Execute Google API call with retry logic"""
    manager = get_ai_key_manager()
    return manager.execute_api_call('google', api_call_func, *args, **kwargs)