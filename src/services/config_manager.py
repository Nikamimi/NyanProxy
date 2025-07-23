"""
[CAT] NyanProxy Configuration Manager with Firebase Persistence
==========================================================

This module provides persistent configuration storage using Firebase Realtime Database.
Settings are stored under configs/ path in Firebase for easy management and persistence
across server restarts.
"""

import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: Firebase not available. Config will use local storage only.")

@dataclass
class AntiAbuseConfig:
    """Anti-abuse/anti-hairball configuration settings"""
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    max_ips_per_user: int = 3
    max_ips_auto_ban: bool = True
    last_updated: str = None
    updated_by: str = "system"
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

class ConfigManager:
    """Firebase-based configuration manager for persistent settings"""
    
    def __init__(self, firebase_path: str = "configs"):
        self.firebase_path = firebase_path
        self.db_ref = None
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes cache timeout
        self.cache_timestamps = {}
        self.lock = threading.Lock()
        
        if FIREBASE_AVAILABLE:
            try:
                # Initialize Firebase if not already done
                if not firebase_admin._apps:
                    from ..config.auth import auth_config
                    if auth_config.firebase_service_account_key and auth_config.firebase_url:
                        # Parse service account key
                        if isinstance(auth_config.firebase_service_account_key, str):
                            service_account_info = json.loads(auth_config.firebase_service_account_key)
                        else:
                            service_account_info = auth_config.firebase_service_account_key
                        
                        cred = credentials.Certificate(service_account_info)
                        firebase_admin.initialize_app(cred, {
                            'databaseURL': auth_config.firebase_url
                        })
                
                self.db_ref = db.reference(self.firebase_path)
                self.firebase_enabled = True
                print(f"[CAT] ConfigManager: Firebase initialized at path '{self.firebase_path}'")
            except Exception as e:
                print(f"[ERROR] ConfigManager: Firebase initialization failed: {e}")
                self.firebase_enabled = False
        else:
            self.firebase_enabled = False
            print("[CAT] ConfigManager: Running in local-only mode")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached value is still valid"""
        if key not in self.cache_timestamps:
            return False
        return (time.time() - self.cache_timestamps[key]) < self.cache_timeout
    
    def _update_cache(self, key: str, value: Any):
        """Update cache with new value and timestamp"""
        with self.lock:
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
    
    def save_anti_abuse_config(self, config: AntiAbuseConfig) -> bool:
        """Save anti-abuse configuration to Firebase"""
        try:
            config.last_updated = datetime.now().isoformat()
            config_dict = asdict(config)
            
            if self.firebase_enabled and self.db_ref:
                # Save to Firebase
                self.db_ref.child('anti_abuse').set(config_dict)
                print(f"[CAT] ConfigManager: Anti-abuse config saved to Firebase")
            
            # Update local cache
            self._update_cache('anti_abuse', config_dict)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] ConfigManager: Failed to save anti-abuse config: {e}")
            return False
    
    def load_anti_abuse_config(self) -> AntiAbuseConfig:
        """Load anti-abuse configuration from Firebase or cache"""
        try:
            # Check cache first
            if self._is_cache_valid('anti_abuse'):
                config_dict = self.cache['anti_abuse']
                return AntiAbuseConfig(**config_dict)
            
            if self.firebase_enabled and self.db_ref:
                # Load from Firebase
                config_data = self.db_ref.child('anti_abuse').get()
                if config_data:
                    self._update_cache('anti_abuse', config_data)
                    return AntiAbuseConfig(**config_data)
            
            # Return default config if nothing found
            default_config = AntiAbuseConfig()
            self._update_cache('anti_abuse', asdict(default_config))
            return default_config
            
        except Exception as e:
            print(f"[ERROR] ConfigManager: Failed to load anti-abuse config: {e}")
            # Return default config on error
            return AntiAbuseConfig()
    
    def save_general_config(self, key: str, value: Any, updated_by: str = "system") -> bool:
        """Save any general configuration value"""
        try:
            config_data = {
                'value': value,
                'last_updated': datetime.now().isoformat(),
                'updated_by': updated_by
            }
            
            if self.firebase_enabled and self.db_ref:
                self.db_ref.child('general').child(key).set(config_data)
                print(f"[CAT] ConfigManager: Config '{key}' saved to Firebase")
            
            # Update local cache
            self._update_cache(f'general_{key}', config_data)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] ConfigManager: Failed to save config '{key}': {e}")
            return False
    
    def load_general_config(self, key: str, default_value: Any = None) -> Any:
        """Load any general configuration value"""
        try:
            cache_key = f'general_{key}'
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['value']
            
            if self.firebase_enabled and self.db_ref:
                # Load from Firebase
                config_data = self.db_ref.child('general').child(key).get()
                if config_data and 'value' in config_data:
                    self._update_cache(cache_key, config_data)
                    return config_data['value']
            
            # Return default if nothing found
            if default_value is not None:
                default_data = {
                    'value': default_value,
                    'last_updated': datetime.now().isoformat(),
                    'updated_by': 'system'
                }
                self._update_cache(cache_key, default_data)
            
            return default_value
            
        except Exception as e:
            print(f"[ERROR] ConfigManager: Failed to load config '{key}': {e}")
            return default_value
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations from Firebase"""
        try:
            if self.firebase_enabled and self.db_ref:
                all_configs = self.db_ref.get()
                return all_configs or {}
            return {}
        except Exception as e:
            print(f"[ERROR] ConfigManager: Failed to get all configs: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached configurations"""
        with self.lock:
            self.cache.clear()
            self.cache_timestamps.clear()
        print("[CAT] ConfigManager: Cache cleared")

# Global config manager instance
config_manager = ConfigManager()