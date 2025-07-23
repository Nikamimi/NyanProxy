"""
Key Pool Management System with Health Monitoring and Real-time Updates
"""
import os
import time
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

try:
    import firebase_admin
    from firebase_admin import db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from .health_checker import HealthCheckManager, HealthResult


class KeyStatus(Enum):
    HEALTHY = "healthy"
    INVALID = "invalid_key"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    TEMPORARILY_DISABLED = "temporarily_disabled"


@dataclass
class APIKey:
    """Represents an API key with its status and metadata"""
    key: str
    provider: str
    status: KeyStatus = KeyStatus.HEALTHY
    last_checked: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    response_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Mask the key for logging purposes
        self.masked_key = self.key[:8] + "..." + self.key[-4:] if len(self.key) > 12 else "***"
    
    def mark_success(self, response_time: float = None):
        """Mark key as successfully used"""
        self.success_count += 1
        self.last_used = datetime.now()
        if response_time:
            self.response_time = response_time
        
        # Reset status to healthy if it was temporarily disabled
        if self.status == KeyStatus.TEMPORARILY_DISABLED:
            self.status = KeyStatus.HEALTHY
    
    def mark_error(self, error_status: str, error_message: str = None):
        """Mark key with an error"""
        self.error_count += 1
        self.last_error = error_message
        self.last_checked = datetime.now()
        
        # Update status based on error type
        if error_status == "invalid_key":
            self.status = KeyStatus.INVALID
        elif error_status == "quota_exceeded":
            self.status = KeyStatus.QUOTA_EXCEEDED
        elif error_status == "rate_limited":
            self.status = KeyStatus.RATE_LIMITED
        else:
            # For other errors, temporarily disable
            self.status = KeyStatus.TEMPORARILY_DISABLED
    
    def should_be_removed(self) -> bool:
        """Check if key should be permanently removed from pool"""
        return self.status in [KeyStatus.INVALID, KeyStatus.QUOTA_EXCEEDED]
    
    def is_usable(self) -> bool:
        """Check if key can be used for requests"""
        return self.status == KeyStatus.HEALTHY
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage"""
        total_requests = self.success_count + self.error_count
        if total_requests == 0:
            return 100.0
        return (self.success_count / total_requests) * 100


class KeyPoolManager:
    """
    Manages API key pools with health monitoring and real-time updates
    """
    
    def __init__(self, firebase_db=None):
        self.lock = threading.Lock()
        self.health_manager = HealthCheckManager()
        self.firebase_db = firebase_db
        
        # Key pools organized by provider
        self.key_pools: Dict[str, List[APIKey]] = {}
        self.removed_keys: Dict[str, List[APIKey]] = {}  # Track removed keys for dashboard
        
        # Configuration
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL_MINUTES', '10'))
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_failures': 0,
            'keys_removed': 0,
            'last_health_check': None
        }
        
        # Start background health monitoring
        self._start_health_monitoring()
        
        print(f"[KEY] Key Pool Manager initialized with MAX_RETRIES={self.max_retries}")
    
    def _start_health_monitoring(self):
        """Start background thread for periodic health checks"""
        def health_monitor():
            while True:
                try:
                    time.sleep(self.health_check_interval * 60)  # Convert minutes to seconds
                    self.perform_health_checks()
                except Exception as e:
                    logging.error(f"Error in health monitoring: {e}")
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
        print(f"[HEALTH] Health monitoring started (every {self.health_check_interval} minutes)")
    
    def add_keys(self, provider: str, keys: List[str]) -> int:
        """Add API keys to the pool for a provider"""
        with self.lock:
            if provider not in self.key_pools:
                self.key_pools[provider] = []
                self.removed_keys[provider] = []
            
            added_count = 0
            for key in keys:
                # Check if key already exists
                existing_keys = [k.key for k in self.key_pools[provider]]
                if key not in existing_keys:
                    api_key = APIKey(key=key, provider=provider)
                    self.key_pools[provider].append(api_key)
                    added_count += 1
                    print(f"[KEY] Added key {api_key.masked_key} to {provider} pool")
            
            print(f"[KEY] Added {added_count} new keys to {provider} pool (total: {len(self.key_pools[provider])})")
            return added_count
    
    def get_healthy_key(self, provider: str) -> Optional[APIKey]:
        """Get a healthy API key for the specified provider"""
        with self.lock:
            if provider not in self.key_pools:
                return None
            
            # Filter for healthy keys
            healthy_keys = [k for k in self.key_pools[provider] if k.is_usable()]
            
            if not healthy_keys:
                print(f"[WARN] No healthy keys available for {provider}")
                return None
            
            # Return the key with best success rate and least recent usage
            best_key = min(healthy_keys, key=lambda k: (
                -k.get_success_rate(),  # Higher success rate first
                k.last_used or datetime.min  # Less recently used first
            ))
            
            return best_key
    
    def mark_key_result(self, provider: str, key: str, success: bool, 
                       error_status: str = None, error_message: str = None, 
                       response_time: float = None):
        """Mark the result of using a key and update its status"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if provider not in self.key_pools:
                return
            
            # Find the key in the pool
            api_key = None
            for k in self.key_pools[provider]:
                if k.key == key:
                    api_key = k
                    break
            
            if not api_key:
                print(f"[WARN] Key not found in pool: {key[:8]}...")
                return
            
            if success:
                api_key.mark_success(response_time)
                print(f"[OK] Key {api_key.masked_key} successful (success rate: {api_key.get_success_rate():.1f}%)")
            else:
                self.stats['total_failures'] += 1
                api_key.mark_error(error_status, error_message)
                print(f"[FAIL] Key {api_key.masked_key} failed: {error_status} - {error_message}")
                
                # Remove key if it should be permanently removed
                if api_key.should_be_removed():
                    self._remove_key_from_pool(provider, api_key)
            
            # Save updated stats to Firebase
            self._save_stats_async()
    
    def _remove_key_from_pool(self, provider: str, api_key: APIKey):
        """Remove a key from the active pool and track it"""
        try:
            self.key_pools[provider].remove(api_key)
            self.removed_keys[provider].append(api_key)
            self.stats['keys_removed'] += 1
            
            print(f"[DELETE] REMOVED key {api_key.masked_key} from {provider} pool (reason: {api_key.status.value})")
            print(f"[KEY] {provider} pool now has {len(self.key_pools[provider])} healthy keys")
            
            # Save removal to Firebase for dashboard visibility
            self._save_removed_key_async(provider, api_key)
            
        except ValueError:
            # Key already removed
            pass
    
    def perform_health_checks(self):
        """Perform health checks on all keys"""
        print(f"[HEALTH] Starting health checks at {datetime.now()}")
        
        with self.lock:
            total_checked = 0
            total_removed = 0
            
            for provider, keys in self.key_pools.items():
                print(f"[HEALTH] Checking {len(keys)} keys for {provider}")
                
                keys_to_check = keys.copy()  # Create copy to avoid modification during iteration
                
                for api_key in keys_to_check:
                    try:
                        result = self.health_manager.check_service_health(provider, api_key.key)
                        total_checked += 1
                        
                        api_key.last_checked = datetime.now()
                        api_key.response_time = result.response_time
                        
                        if result.status == 'healthy':
                            api_key.status = KeyStatus.HEALTHY
                            print(f"🟢 {api_key.masked_key} healthy ({result.response_time:.2f}s)")
                        else:
                            # Update key status based on health check result
                            if result.status == 'invalid_key':
                                api_key.status = KeyStatus.INVALID
                            elif result.status == 'quota_exceeded':
                                api_key.status = KeyStatus.QUOTA_EXCEEDED
                            elif result.status == 'rate_limited':
                                api_key.status = KeyStatus.RATE_LIMITED
                            else:
                                api_key.status = KeyStatus.TEMPORARILY_DISABLED
                            
                            api_key.last_error = result.error_message
                            print(f"[RED] {api_key.masked_key} {result.status}: {result.error_message}")
                            
                            # Remove if permanently failed
                            if api_key.should_be_removed():
                                self._remove_key_from_pool(provider, api_key)
                                total_removed += 1
                    
                    except Exception as e:
                        print(f"[FAIL] Health check error for {api_key.masked_key}: {e}")
                        api_key.status = KeyStatus.TEMPORARILY_DISABLED
                        api_key.last_error = str(e)
            
            self.stats['last_health_check'] = datetime.now()
            print(f"[HEALTH] Health check complete: {total_checked} checked, {total_removed} removed")
    
    def get_pool_status(self, provider: str = None) -> Dict:
        """Get status of key pools"""
        with self.lock:
            if provider:
                # Status for specific provider
                if provider not in self.key_pools:
                    return {'error': f'Provider {provider} not found'}
                
                keys = self.key_pools[provider]
                status_counts = {}
                for status in KeyStatus:
                    status_counts[status.value] = len([k for k in keys if k.status == status])
                
                return {
                    'provider': provider,
                    'total_keys': len(keys),
                    'healthy_keys': len([k for k in keys if k.is_usable()]),
                    'status_breakdown': status_counts,
                    'removed_keys': len(self.removed_keys.get(provider, [])),
                    'keys': [
                        {
                            'masked_key': k.masked_key,
                            'status': k.status.value,
                            'success_rate': k.get_success_rate(),
                            'last_used': k.last_used.isoformat() if k.last_used else None,
                            'last_checked': k.last_checked.isoformat() if k.last_checked else None,
                            'response_time': k.response_time,
                            'error_count': k.error_count,
                            'success_count': k.success_count
                        } for k in keys
                    ]
                }
            else:
                # Status for all providers
                result = {
                    'global_stats': self.stats.copy(),
                    'providers': {}
                }
                
                if result['global_stats']['last_health_check']:
                    result['global_stats']['last_health_check'] = result['global_stats']['last_health_check'].isoformat()
                
                for prov in self.key_pools.keys():
                    result['providers'][prov] = self.get_pool_status(prov)
                
                return result
    
    def get_removed_keys(self, provider: str = None) -> Dict:
        """Get information about removed keys"""
        with self.lock:
            if provider:
                removed = self.removed_keys.get(provider, [])
                return {
                    'provider': provider,
                    'count': len(removed),
                    'keys': [
                        {
                            'masked_key': k.masked_key,
                            'status': k.status.value,
                            'last_error': k.last_error,
                            'error_count': k.error_count,
                            'success_count': k.success_count,
                            'removed_at': k.last_checked.isoformat() if k.last_checked else None
                        } for k in removed
                    ]
                }
            else:
                result = {}
                for prov in self.removed_keys.keys():
                    result[prov] = self.get_removed_keys(prov)
                return result
    
    def _save_stats_async(self):
        """Save statistics to Firebase asynchronously"""
        if not self.firebase_db:
            return
        
        def save_stats():
            try:
                stats_data = self.stats.copy()
                if stats_data['last_health_check']:
                    stats_data['last_health_check'] = stats_data['last_health_check'].isoformat()
                
                self.firebase_db.child('key_pool_stats').set(stats_data)
            except Exception as e:
                print(f"Failed to save key pool stats: {e}")
        
        thread = threading.Thread(target=save_stats)
        thread.daemon = True
        thread.start()
    
    def _save_removed_key_async(self, provider: str, api_key: APIKey):
        """Save removed key info to Firebase asynchronously"""
        if not self.firebase_db:
            return
        
        def save_removed_key():
            try:
                key_data = {
                    'masked_key': api_key.masked_key,
                    'status': api_key.status.value,
                    'last_error': api_key.last_error,
                    'error_count': api_key.error_count,
                    'success_count': api_key.success_count,
                    'removed_at': datetime.now().isoformat(),
                    'success_rate': api_key.get_success_rate()
                }
                
                self.firebase_db.child(f'removed_keys/{provider}').push(key_data)
            except Exception as e:
                print(f"Failed to save removed key data: {e}")
        
        thread = threading.Thread(target=save_removed_key)
        thread.daemon = True
        thread.start()
    
    def force_remove_key(self, provider: str, key_identifier: str) -> bool:
        """Manually remove a key from the pool"""
        with self.lock:
            if provider not in self.key_pools:
                return False
            
            for api_key in self.key_pools[provider]:
                if api_key.key == key_identifier or api_key.masked_key == key_identifier:
                    api_key.status = KeyStatus.INVALID
                    api_key.last_error = "Manually removed"
                    self._remove_key_from_pool(provider, api_key)
                    print(f"[TOOL] Manually removed key {api_key.masked_key} from {provider} pool")
                    return True
            
            return False
    
    def get_max_retries(self) -> int:
        """Get the maximum number of retries configured"""
        return self.max_retries
    
    def set_max_retries(self, retries: int):
        """Set the maximum number of retries"""
        self.max_retries = max(1, retries)
        print(f"[TOOL] Max retries set to {self.max_retries}")