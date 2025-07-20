"""
🐱 Model Families System for NyanProxy
=====================================

This module provides comprehensive model management with:
- Whitelisting specific models for each AI service
- Individual model usage tracking
- Cost calculation per model
- Cat-themed model categorization
"""

import json
import threading
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import os

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from ..config.auth import auth_config

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    GROQ = "groq"
    COHERE = "cohere"

@dataclass
class ModelInfo:
    """Information about a specific AI model"""
    model_id: str
    provider: AIProvider
    display_name: str
    description: str
    input_cost_per_1m: float  # Cost per 1 million input tokens
    output_cost_per_1m: float  # Cost per 1 million output tokens
    context_length: int
    supports_streaming: bool = True
    supports_function_calling: bool = False
    is_premium: bool = False
    cat_personality: str = "curious"  # curious, playful, sleepy, grumpy
    max_input_tokens: Optional[int] = None  # Maximum allowed input tokens
    max_output_tokens: Optional[int] = None  # Maximum allowed output tokens
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = (input_tokens / 1000000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1000000) * self.output_cost_per_1m
        return input_cost + output_cost
    
    def get_cat_emoji(self) -> str:
        """Get cat emoji based on personality"""
        personalities = {
            "curious": "🙀",
            "playful": "😸",
            "sleepy": "😴",
            "grumpy": "😾",
            "smart": "🤓",
            "fast": "💨"
        }
        return personalities.get(self.cat_personality, "😺")
    
    def validate_token_limits(self, input_tokens: int, output_tokens: int = None) -> tuple[bool, str]:
        """Validate if request respects token limits"""
        # Check input tokens
        if self.max_input_tokens and input_tokens > self.max_input_tokens:
            return False, f"Input token limit exceeded. Maximum allowed: {self.max_input_tokens:,}, provided: {input_tokens:,}"
        
        # Check output tokens if specified
        if output_tokens and self.max_output_tokens and output_tokens > self.max_output_tokens:
            return False, f"Output token limit exceeded. Maximum allowed: {self.max_output_tokens:,}, requested: {output_tokens:,}"
        
        return True, ""

@dataclass
class ModelUsageStats:
    """Usage statistics for a specific model"""
    model_id: str
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 100.0
    
    def add_usage(self, input_tokens: int, output_tokens: int, cost: float, success: bool = True):
        """Add usage data"""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        
        now = datetime.now()
        if self.first_used is None:
            self.first_used = now
        self.last_used = now
        
        if not success:
            self.error_count += 1
        
        # Update success rate
        self.success_rate = ((self.total_requests - self.error_count) / self.total_requests) * 100
    
    def to_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'total_requests': self.total_requests,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': self.total_cost,
            'first_used': self.first_used.isoformat() if self.first_used else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'error_count': self.error_count,
            'success_rate': self.success_rate
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            model_id=data['model_id'],
            total_requests=data.get('total_requests', 0),
            total_input_tokens=data.get('total_input_tokens', 0),
            total_output_tokens=data.get('total_output_tokens', 0),
            total_cost=data.get('total_cost', 0.0),
            first_used=datetime.fromisoformat(data['first_used']) if data.get('first_used') else None,
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            error_count=data.get('error_count', 0),
            success_rate=data.get('success_rate', 100.0)
        )

class ModelFamilyManager:
    """
    🐾 Model Family Management System
    
    Manages whitelisted models for each AI service and tracks individual usage.
    """
    
    @staticmethod
    def sanitize_key(key: str) -> str:
        """
        Sanitize model ID for use as Firebase key
        Replaces problematic characters with safe alternatives
        """
        return (key
                .replace('.', '_')
                .replace('/', '__SLASH__')
                .replace('#', '__HASH__')
                .replace('$', '__DOLLAR__')
                .replace('[', '__LBRACKET__')
                .replace(']', '__RBRACKET__'))
    
    @staticmethod
    def unsanitize_key(key: str) -> str:
        """
        Convert sanitized key back to original model ID
        """
        # First handle the special encoded sequences
        result = (key
                .replace('__SLASH__', '/')
                .replace('__HASH__', '#')
                .replace('__DOLLAR__', '$')
                .replace('__LBRACKET__', '[')
                .replace('__RBRACKET__', ']'))
        
        # For periods, we need to be careful - only replace single underscores that represent periods
        # This is a simple heuristic: replace underscores with periods only in version-like patterns
        import re
        # Replace patterns like gpt-4_1 back to gpt-4.1
        result = re.sub(r'(\d)_(\d)', r'\1.\2', result)
        
        return result
    
    def __init__(self):
        self.lock = threading.Lock()
        self.models: Dict[str, ModelInfo] = {}
        self.whitelisted_models: Dict[AIProvider, Set[str]] = {}
        self.global_usage_stats: Dict[str, ModelUsageStats] = {}
        self.user_usage_stats: Dict[str, Dict[str, ModelUsageStats]] = {}  # user_token -> model_id -> stats
        self.global_totals = {
            'total_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'total_models_used': 0,
            'first_request': None,
            'last_request': None
        }
        self.firebase_db = None
        self._last_save_time = datetime.now()  # Track last save time
        # Save interval configurable via environment variable, default 1 minute (60 seconds)
        self._save_interval_minutes = int(os.getenv('GLOBAL_STATS_SAVE_INTERVAL_MINUTES', 1))
        print(f"🐱 Global stats will be saved to Firebase every {self._save_interval_minutes} minutes")
        
        # Initialize with default models
        self._initialize_default_models()
        
        # Initialize Firebase if available
        if FIREBASE_AVAILABLE and auth_config.firebase_url:
            self._initialize_firebase()
        
        # Load configuration
        self._load_configuration()
        
        # Load custom models
        self._load_custom_models()
        
        # Load model overrides (modifications to default models)
        self._load_model_overrides()
        
        # Load global totals from Firebase
        self._load_global_totals()
        
        # Load global usage stats from Firebase
        self._load_global_usage_stats()
        
        # Clean up any duplicate entries in whitelist
        self.clean_whitelist_duplicates()
        
        # Apply default whitelists only if no configuration was loaded
        self._apply_default_whitelists_if_needed()
    
    def _initialize_default_models(self):
        """Initialize with default model configurations"""
        default_models = [
            # OpenAI Models (prices per 1M tokens)
            ModelInfo(
                model_id="gpt-4o",
                provider=AIProvider.OPENAI,
                display_name="GPT-4o",
                description="Most advanced GPT-4 model with vision capabilities",
                input_cost_per_1m=5.0,
                output_cost_per_1m=15.0,
                context_length=128000,
                supports_function_calling=True,
                is_premium=True,
                cat_personality="smart"
            ),
            
            # Anthropic Models (prices per 1M tokens)
            ModelInfo(
                model_id="claude-3-5-sonnet-20241022",
                provider=AIProvider.ANTHROPIC,
                display_name="Claude 3.5 Sonnet",
                description="Most intelligent Claude model",
                input_cost_per_1m=3.0,
                output_cost_per_1m=15.0,
                context_length=200000,
                is_premium=True,
                cat_personality="smart"
            ),
            
            # Google Gemini Models (prices per 1M tokens)
            ModelInfo(
                model_id="gemini-1.5-pro-latest",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 1.5 Pro",
                description="Google's most capable model with 1M token context",
                input_cost_per_1m=1.25,
                output_cost_per_1m=5.0,
                context_length=1000000,
                supports_function_calling=True,
                is_premium=True,
                cat_personality="smart"
            ),
            ModelInfo(
                model_id="gemini-1.5-flash",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 1.5 Flash",
                description="Fast and efficient model for high-volume tasks",
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.30,
                context_length=1000000,
                supports_function_calling=True,
                cat_personality="fast"
            ),
            ModelInfo(
                model_id="gemini-2.0-flash-exp",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 2.0 Flash (Experimental)",
                description="Latest experimental Gemini model with enhanced capabilities",
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.30,
                context_length=1000000,
                supports_function_calling=True,
                is_premium=True,
                cat_personality="playful"
            ),
            ModelInfo(
                model_id="gemini-2.5-flash",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 2.5 Flash",
                description="Latest Gemini 2.5 Flash model with improved performance",
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.30,
                context_length=1000000,
                supports_function_calling=True,
                is_premium=True,
                cat_personality="fast"
            ),
            ModelInfo(
                model_id="gemini-2.5-pro",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 2.5 Pro",
                description="Advanced Gemini 2.5 Pro model with enhanced capabilities",
                input_cost_per_1m=1.25,
                output_cost_per_1m=5.0,
                context_length=2000000,
                supports_function_calling=True,
                is_premium=True,
                cat_personality="smart"
            ),
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
        
        # Initialize empty whitelists (will be populated from saved config or defaults if none exist)
        for provider in AIProvider:
            self.whitelisted_models[provider] = set()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection for persistent storage"""
        try:
            # Firebase is already initialized in user_store, just get reference
            self.firebase_db = db.reference()
            # Model families connected to Firebase
            
            # Load existing configuration with timeout
            self._load_from_firebase_safe()
            
        except Exception as e:
            print(f"Failed to connect model families to Firebase: {e}")
            self.firebase_db = None
    
    def _load_configuration(self):
        """Load configuration from local file if Firebase not available"""
        if self.firebase_db:
            return  # Firebase handles persistence
        
        config_file = "model_families_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Load whitelisted models
                for provider_str, model_list in config.get('whitelisted_models', {}).items():
                    try:
                        provider = AIProvider(provider_str)
                        # Clean the list: remove None values and duplicates
                        clean_list = [mid for mid in model_list if mid is not None]
                        self.whitelisted_models[provider] = set(clean_list)
                        print(f"📥 Loaded {len(clean_list)} models for {provider_str}: {sorted(clean_list)}")
                    except ValueError:
                        print(f"Unknown provider in config: {provider_str}")
                
                # Loaded model families configuration from file
                
            except Exception as e:
                print(f"Failed to load model families config: {e}")
    
    def _load_from_firebase_safe(self):
        """Load configuration from Firebase with timeout protection"""
        try:
            if not self.firebase_db:
                return
            
            # Use a thread with timeout to prevent blocking
            import threading
            import signal
            
            def load_config():
                try:
                    config_ref = self.firebase_db.child('model_families_config')
                    config = config_ref.get()
                    
                    if config:
                        # Load whitelisted models
                        for provider_str, model_list in config.get('whitelisted_models', {}).items():
                            try:
                                provider = AIProvider(provider_str)
                                # Clean the list: remove None values and duplicates
                                clean_list = [mid for mid in model_list if mid is not None]
                                self.whitelisted_models[provider] = set(clean_list)
                                print(f"🔥 Firebase loaded {len(clean_list)} models for {provider_str}: {sorted(clean_list)}")
                            except ValueError:
                                print(f"Unknown provider in Firebase config: {provider_str}")
                        
                        print("Loaded model families configuration from Firebase")
                except Exception as e:
                    print(f"Failed to load config in thread: {e}")
            
            # Try to load with a timeout
            config_thread = threading.Thread(target=load_config)
            config_thread.daemon = True
            config_thread.start()
            config_thread.join(timeout=5.0)  # 5 second timeout
            
            if config_thread.is_alive():
                print("Firebase config load timed out, using defaults")
                
        except Exception as e:
            print(f"Failed to load model families from Firebase: {e}")
    
    def _load_from_firebase(self):
        """Legacy method - kept for compatibility"""
        self._load_from_firebase_safe()
    
    def _save_configuration(self):
        """Save configuration to persistent storage"""
        # Ensure no duplicates and filter out None values
        config = {
            'whitelisted_models': {
                provider.value: sorted(list(set(filter(None, models))))
                for provider, models in self.whitelisted_models.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        print(f"💾 Saving configuration to Firebase/file:")
        for provider, models in config['whitelisted_models'].items():
            print(f"💾   {provider}: {models}")
        
        if self.firebase_db:
            try:
                # Use thread with timeout for Firebase saves
                import threading
                
                def save_to_firebase():
                    try:
                        config_ref = self.firebase_db.child('model_families_config')
                        config_ref.set(config)
                        print("Saved model families config to Firebase")
                    except Exception as e:
                        print(f"Failed to save in thread: {e}")
                
                save_thread = threading.Thread(target=save_to_firebase)
                save_thread.daemon = True
                save_thread.start()
                save_thread.join(timeout=3.0)  # 3 second timeout
                
                if save_thread.is_alive():
                    print("Firebase save timed out")
                    
            except Exception as e:
                print(f"Failed to save model families to Firebase: {e}")
        else:
            try:
                import os
                os.makedirs("data", exist_ok=True)
                with open("data/model_families_config.json", 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                print(f"Failed to save model families to file: {e}")
    
    def _save_configuration_sync(self):
        """Save configuration synchronously (for critical operations like model addition)"""
        # Ensure no duplicates and filter out None values
        config = {
            'whitelisted_models': {
                provider.value: sorted(list(set(filter(None, models))))
                for provider, models in self.whitelisted_models.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        if self.firebase_db:
            try:
                import threading
                success = [False]
                
                def save_config():
                    try:
                        config_ref = self.firebase_db.child('model_families_config')
                        config_ref.set(config)
                        success[0] = True
                        print("Saved model families config to Firebase (sync)")
                    except Exception as e:
                        print(f"Failed to save model families to Firebase (sync): {e}")
                
                # Run with timeout protection
                save_thread = threading.Thread(target=save_config)
                save_thread.daemon = True
                save_thread.start()
                save_thread.join(timeout=5.0)  # 5 second timeout
                
                if save_thread.is_alive():
                    print("Firebase config save timed out, falling back to local file")
                    
            except Exception as e:
                print(f"Failed to save model families to Firebase (sync): {e}")
        
        # Always save to local file as backup
        try:
            import os
            os.makedirs("data", exist_ok=True)
            with open("data/model_families_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save model families to file: {e}")
    
    def is_model_whitelisted(self, provider: AIProvider, model_id: str) -> bool:
        """Check if a model is whitelisted for use"""
        with self.lock:
            try:
                whitelisted = self.whitelisted_models.get(provider, set())
                is_whitelisted = model_id in whitelisted
                
                if not is_whitelisted:
                    # Check if any whitelisted model has this as its actual model_id
                    for whitelisted_id in whitelisted:
                        stored_model = self.models.get(whitelisted_id)
                        if stored_model and stored_model.model_id == model_id:
                            return True
                
                return is_whitelisted
            except Exception as e:
                import logging
                logging.error(f"Error checking whitelist for model '{model_id}': {e}")
                return False
    
    def get_whitelisted_models(self, provider: AIProvider) -> List[ModelInfo]:
        """Get all whitelisted models for a provider"""
        with self.lock:
            whitelisted_ids = self.whitelisted_models.get(provider, set())
            return [
                self.models[model_id] 
                for model_id in whitelisted_ids 
                if model_id in self.models
            ]
    
    def get_all_models(self, provider: Optional[AIProvider] = None) -> List[ModelInfo]:
        """Get all available models, optionally filtered by provider"""
        with self.lock:
            if provider:
                return [model for model in self.models.values() if model.provider == provider]
            return list(self.models.values())
    
    def add_model_to_whitelist(self, provider: AIProvider, model_id: str) -> bool:
        """Add a model to the whitelist"""
        with self.lock:
            if model_id not in self.models:
                print(f"❌ Cannot add {model_id} to whitelist: model not found")
                return False
            
            if provider not in self.whitelisted_models:
                self.whitelisted_models[provider] = set()
            
            print(f"🟢 Adding {model_id} to {provider.value} whitelist")
            print(f"🟢 Before: {sorted(self.whitelisted_models[provider])}")
            self.whitelisted_models[provider].add(model_id)
            print(f"🟢 After: {sorted(self.whitelisted_models[provider])}")
            self._save_configuration()
            return True
    
    def remove_model_from_whitelist(self, provider: AIProvider, model_id: str) -> bool:
        """Remove a model from the whitelist"""
        with self.lock:
            if provider not in self.whitelisted_models:
                return False
            
            if model_id in self.whitelisted_models[provider]:
                self.whitelisted_models[provider].remove(model_id)
                self._save_configuration()
                return True
            
            return False
    
    def set_provider_whitelist(self, provider: AIProvider, model_ids: List[str]) -> bool:
        """Set the complete whitelist for a provider"""
        with self.lock:
            # Filter out None values and validate all model IDs exist
            clean_ids = [mid for mid in model_ids if mid is not None and mid in self.models]
            # Remove duplicates by converting to set
            unique_ids = set(clean_ids)
            
            print(f"🔧 Setting whitelist for {provider.value}: {sorted(unique_ids)}")
            self.whitelisted_models[provider] = unique_ids
            self._save_configuration()
            return len(unique_ids) == len([mid for mid in model_ids if mid is not None])
    
    def clean_whitelist_duplicates(self):
        """Clean up any duplicate entries in the whitelist and save"""
        with self.lock:
            cleaned = False
            for provider, models in self.whitelisted_models.items():
                original_count = len(models)
                # Convert to list, filter None, remove duplicates, convert back to set
                clean_models = set(filter(None, models))
                if len(clean_models) != original_count:
                    print(f"🧹 Cleaned {provider.value}: {original_count} -> {len(clean_models)} models")
                    self.whitelisted_models[provider] = clean_models
                    cleaned = True
            
            if cleaned:
                print("🧹 Saving cleaned whitelist configuration...")
                self._save_configuration_sync()  # Use sync to ensure it's saved
            
            return cleaned
    
    def _apply_default_whitelists_if_needed(self):
        """Apply default model whitelists only if no saved configuration exists"""
        with self.lock:
            # Check if any provider has any models whitelisted
            has_any_config = any(
                len(models) > 0 for models in self.whitelisted_models.values()
            )
            
            if not has_any_config:
                print("🔧 No existing whitelist configuration found, applying defaults...")
                # Get default models by provider
                default_models_by_provider = {}
                for model_id, model in self.models.items():
                    # Only include truly essential default models, not all of them
                    essential_defaults = {
                        "gpt-4o-mini",  # Cheap OpenAI model
                        "gemini-1.5-flash",  # Cheap Google model
                        # Don't auto-whitelist expensive Anthropic models
                    }
                    if model_id in essential_defaults:
                        provider = model.provider
                        if provider not in default_models_by_provider:
                            default_models_by_provider[provider] = set()
                        default_models_by_provider[provider].add(model_id)
                
                # Apply minimal defaults
                for provider, model_ids in default_models_by_provider.items():
                    self.whitelisted_models[provider] = model_ids
                    print(f"🔧 Applied default whitelist for {provider.value}: {sorted(model_ids)}")
                
                # Save the minimal defaults
                if default_models_by_provider:
                    self._save_configuration()
            else:
                print("✅ Existing whitelist configuration found, using saved preferences")
    
    def track_model_usage(self, user_token: str, model_id: str, input_tokens: int, 
                         output_tokens: int, success: bool = True) -> Optional[float]:
        """Track usage for a specific model and return cost"""
        with self.lock:
            if model_id not in self.models:
                return None
            
            model_info = self.models[model_id]
            cost = model_info.calculate_cost(input_tokens, output_tokens)
            
            # Update global stats
            if model_id not in self.global_usage_stats:
                self.global_usage_stats[model_id] = ModelUsageStats(model_id=model_id)
            
            self.global_usage_stats[model_id].add_usage(input_tokens, output_tokens, cost, success)
            
            # Update user-specific stats
            if user_token not in self.user_usage_stats:
                self.user_usage_stats[user_token] = {}
            
            if model_id not in self.user_usage_stats[user_token]:
                self.user_usage_stats[user_token][model_id] = ModelUsageStats(model_id=model_id)
            
            self.user_usage_stats[user_token][model_id].add_usage(input_tokens, output_tokens, cost, success)
            
            # Update global totals
            now = datetime.now()
            self.global_totals['total_requests'] += 1
            self.global_totals['total_input_tokens'] += input_tokens
            self.global_totals['total_output_tokens'] += output_tokens
            self.global_totals['total_cost'] += cost
            
            if self.global_totals['first_request'] is None:
                self.global_totals['first_request'] = now
            self.global_totals['last_request'] = now
            
            # Update unique models count
            unique_models = set(self.global_usage_stats.keys())
            self.global_totals['total_models_used'] = len(unique_models)
            
            # Save to Firebase periodically for better persistence (time-based)
            now = datetime.now()
            time_since_last_save = (now - self._last_save_time).total_seconds() / 60  # Convert to minutes
            
            if time_since_last_save >= self._save_interval_minutes:
                print(f"🐱 Saving global stats to Firebase (last saved {time_since_last_save:.1f} minutes ago)")
                self._save_global_totals()
                self._save_global_usage_stats()
                self._last_save_time = now
            
            return cost
    
    def set_save_interval(self, minutes: int):
        """Set the save interval in minutes"""
        self._save_interval_minutes = max(1, minutes)  # Minimum 1 minute
        print(f"🐱 Global stats save interval set to {self._save_interval_minutes} minutes")
    
    def force_save_global_stats(self):
        """Force immediate save of global statistics"""
        print("🐱 Force saving global stats to Firebase")
        self._save_global_totals()
        self._save_global_usage_stats()
        self._last_save_time = datetime.now()
    
    def _load_global_totals(self):
        """Load global totals from Firebase"""
        if not self.firebase_db:
            return
            
        try:
            import threading
            loaded_totals = [None]
            
            def load_totals():
                try:
                    totals_ref = self.firebase_db.child('global_totals')
                    data = totals_ref.get()
                    
                    if data:
                        # Convert ISO strings back to datetime objects
                        if data.get('first_request'):
                            try:
                                data['first_request'] = datetime.fromisoformat(data['first_request'].replace('Z', '+00:00'))
                            except:
                                data['first_request'] = None
                        if data.get('last_request'):
                            try:
                                data['last_request'] = datetime.fromisoformat(data['last_request'].replace('Z', '+00:00'))
                            except:
                                data['last_request'] = None
                        
                        loaded_totals[0] = data
                        print(f"🐱 Loaded global totals from Firebase: {data.get('total_requests', 0)} requests")
                except Exception as e:
                    print(f"Failed to load global totals from Firebase: {e}")
            
            thread = threading.Thread(target=load_totals)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5)
            
            if loaded_totals[0]:
                with self.lock:
                    self.global_totals.update(loaded_totals[0])
                    
        except Exception as e:
            print(f"Error loading global totals: {e}")
    
    def _save_global_totals(self):
        """Save global totals to Firebase (async)"""
        if not self.firebase_db:
            return
            
        try:
            import threading
            
            def save_totals():
                try:
                    with self.lock:
                        totals = self.global_totals.copy()
                    
                    # Convert datetime objects to ISO strings for Firebase
                    if totals['first_request']:
                        totals['first_request'] = totals['first_request'].isoformat()
                    if totals['last_request']:
                        totals['last_request'] = totals['last_request'].isoformat()
                    
                    totals_ref = self.firebase_db.child('global_totals')
                    totals_ref.set(totals)
                    print(f"🐱 Saved global totals to Firebase: {totals.get('total_requests', 0)} requests")
                except Exception as e:
                    print(f"Failed to save global totals to Firebase: {e}")
            
            thread = threading.Thread(target=save_totals)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"Error saving global totals: {e}")
    
    def _load_global_usage_stats(self):
        """Load global usage stats from Firebase"""
        if not self.firebase_db:
            return
            
        try:
            import threading
            loaded_stats = [None]
            
            def load_stats():
                try:
                    stats_ref = self.firebase_db.child('global_usage_stats')
                    data = stats_ref.get()
                    
                    if data:
                        # Convert data back to ModelUsageStats objects
                        for sanitized_model_id, stats_data in data.items():
                            if isinstance(stats_data, dict) and 'model_id' in stats_data:
                                # Unsanitize the model ID
                                model_id = self.unsanitize_key(sanitized_model_id)
                                # Create ModelUsageStats object from saved data
                                usage_stats = ModelUsageStats.from_dict(stats_data)
                                loaded_stats[0] = loaded_stats[0] or {}
                                loaded_stats[0][model_id] = usage_stats
                        
                        if loaded_stats[0]:
                            print(f"🐱 Loaded global usage stats from Firebase: {len(loaded_stats[0])} models")
                except Exception as e:
                    print(f"Failed to load global usage stats from Firebase: {e}")
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=load_stats)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5)
            
            if loaded_stats[0]:
                with self.lock:
                    self.global_usage_stats.update(loaded_stats[0])
                    
        except Exception as e:
            print(f"Error loading global usage stats: {e}")
    
    def _save_global_usage_stats(self):
        """Save global usage stats to Firebase (async)"""
        if not self.firebase_db:
            print("🐱 [DEBUG] No Firebase DB connection, skipping save")
            return
            
        try:
            import threading
            
            def save_stats():
                try:
                    with self.lock:
                        stats_to_save = {}
                        for model_id, usage_stats in self.global_usage_stats.items():
                            # Sanitize model ID for Firebase key
                            sanitized_key = self.sanitize_key(model_id)
                            stats_dict = usage_stats.to_dict()
                            stats_to_save[sanitized_key] = stats_dict
                    
                    stats_ref = self.firebase_db.child('global_usage_stats')
                    stats_ref.set(stats_to_save)
                    print(f"🐱 Saved global usage stats to Firebase: {len(stats_to_save)} models")
                except Exception as e:
                    print(f"Failed to save global usage stats to Firebase: {e}")
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=save_stats)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"Error saving global usage stats: {e}")
    
    def save_usage_stats_sync(self):
        """Save usage stats synchronously (for manual triggers)"""
        if not self.firebase_db:
            return
            
        try:
            import threading
            success = [False]
            
            def save_stats():
                try:
                    with self.lock:
                        stats_to_save = {}
                        for model_id, usage_stats in self.global_usage_stats.items():
                            # Sanitize model ID for Firebase key
                            sanitized_key = self.sanitize_key(model_id)
                            stats_to_save[sanitized_key] = usage_stats.to_dict()
                    
                    stats_ref = self.firebase_db.child('global_usage_stats')
                    stats_ref.set(stats_to_save)
                    success[0] = True
                    print(f"🐱 Saved global usage stats to Firebase (sync): {len(stats_to_save)} models")
                except Exception as e:
                    print(f"Failed to save global usage stats to Firebase (sync): {e}")
            
            thread = threading.Thread(target=save_stats)
            thread.daemon = True
            thread.start()
            thread.join(timeout=10)  # Longer timeout for sync operation
            
            if not success[0]:
                print("Warning: Synchronous save of usage stats may have failed or timed out")
                
        except Exception as e:
            print(f"Error in sync save of global usage stats: {e}")
    
    def shutdown(self):
        """Save all data before shutdown"""
        print("🐱 Saving model usage data before shutdown...")
        try:
            self.save_usage_stats_sync()
            self.force_save_global_stats()  # Use force save for immediate shutdown saving
            print("🐱 Model usage data saved successfully")
        except Exception as e:
            print(f"Error saving data during shutdown: {e}")
    
    def get_global_totals(self) -> Dict[str, Any]:
        """Get global usage totals across all models and users"""
        with self.lock:
            totals = self.global_totals.copy()
            # Convert datetime objects to ISO strings for JSON serialization
            if totals['first_request']:
                totals['first_request'] = totals['first_request'].isoformat()
            if totals['last_request']:
                totals['last_request'] = totals['last_request'].isoformat()
            return totals
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        with self.lock:
            try:
                # First try exact match
                model_info = self.models.get(model_id)
                if model_info:
                    return model_info
                
                # If not found, check if this might be a model that was stored with sanitized key
                # but we need to find it by its original ID
                import logging
                logging.info(f"Model '{model_id}' not found directly, checking all models...")
                for stored_id, stored_model in self.models.items():
                    if stored_model.model_id == model_id:
                        logging.info(f"Found model '{model_id}' stored as '{stored_id}'")
                        return stored_model
                
                logging.warning(f"Model '{model_id}' not found in {len(self.models)} total models")
                return None
            except Exception as e:
                import logging
                logging.error(f"Error in get_model_info for '{model_id}': {e}")
                return None
    
    def get_usage_stats(self, user_token: Optional[str] = None, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics"""
        with self.lock:
            if user_token and model_id:
                # Specific user and model
                if user_token in self.user_usage_stats and model_id in self.user_usage_stats[user_token]:
                    return self.user_usage_stats[user_token][model_id].to_dict()
                return {}
            
            elif user_token:
                # All models for a specific user
                if user_token in self.user_usage_stats:
                    return {
                        model_id: stats.to_dict() 
                        for model_id, stats in self.user_usage_stats[user_token].items()
                    }
                return {}
            
            elif model_id:
                # Specific model across all users
                if model_id in self.global_usage_stats:
                    return self.global_usage_stats[model_id].to_dict()
                return {}
            
            else:
                # Global stats for all models
                return {
                    model_id: stats.to_dict() 
                    for model_id, stats in self.global_usage_stats.items()
                }
    
    def get_cost_analysis(self, user_token: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        with self.lock:
            if user_token:
                # User-specific analysis
                if user_token not in self.user_usage_stats:
                    return {'total_cost': 0, 'by_model': {}, 'by_provider': {}}
                
                user_stats = self.user_usage_stats[user_token]
            else:
                # Global analysis
                user_stats = self.global_usage_stats
            
            total_cost = 0.0
            by_model = {}
            by_provider = {}
            
            for model_id, stats in user_stats.items():
                model_info = self.models.get(model_id)
                if not model_info:
                    continue
                
                provider = model_info.provider.value
                model_cost = stats.total_cost
                
                total_cost += model_cost
                by_model[model_id] = {
                    'cost': model_cost,
                    'requests': stats.total_requests,
                    'display_name': model_info.display_name,
                    'provider': provider
                }
                
                if provider not in by_provider:
                    by_provider[provider] = {'cost': 0, 'requests': 0, 'models': []}
                
                by_provider[provider]['cost'] += model_cost
                by_provider[provider]['requests'] += stats.total_requests
                by_provider[provider]['models'].append(model_id)
            
            return {
                'total_cost': total_cost,
                'by_model': by_model,
                'by_provider': by_provider,
                'currency': 'USD'
            }
    
    def add_custom_model(self, model_info: ModelInfo, auto_whitelist: bool = True) -> bool:
        """Add a custom model to the system"""
        with self.lock:
            # Check if model already exists
            if model_info.model_id in self.models:
                print(f"❌ Model {model_info.model_id} already exists")
                return False
            
            print(f"🚀 Adding custom model: {model_info.model_id} to provider {model_info.provider.value}")
            
            # Add the model to in-memory storage first
            self.models[model_info.model_id] = model_info
            print(f"✅ Added {model_info.model_id} to in-memory storage")
            
            # Auto-whitelist if requested
            if auto_whitelist:
                provider = model_info.provider
                if provider not in self.whitelisted_models:
                    self.whitelisted_models[provider] = set()
                self.whitelisted_models[provider].add(model_info.model_id)
                print(f"✅ Auto-whitelisted {model_info.model_id} for provider {provider.value}")
            
            try:
                # Save synchronously to ensure data persistence before returning success
                print(f"💾 Saving configuration and model data...")
                self._save_configuration_sync()
                print(f"✅ Configuration saved")
                self._save_custom_models_sync()
                print(f"✅ Custom models saved")
                
                print(f"🎉 Successfully added custom model: {model_info.model_id} to provider {model_info.provider.value}")
                return True
            except Exception as e:
                print(f"❌ Failed to save model data: {e}")
                import traceback
                traceback.print_exc()
                
                # Remove from in-memory storage if save failed
                del self.models[model_info.model_id]
                if auto_whitelist and provider in self.whitelisted_models:
                    self.whitelisted_models[provider].discard(model_info.model_id)
                
                return False
    
    def remove_custom_model(self, model_id: str) -> bool:
        """Remove a custom model from the system"""
        with self.lock:
            if model_id not in self.models:
                return False
            
            model_info = self.models[model_id]
            
            # Remove from whitelist first
            provider = model_info.provider
            if provider in self.whitelisted_models:
                self.whitelisted_models[provider].discard(model_id)
            
            # Remove from models
            del self.models[model_id]
            
            # Save configuration
            self._save_configuration()
            
            # Save custom models to Firebase/file
            self._save_custom_models()
            
            return True
    
    def get_custom_models(self) -> List[ModelInfo]:
        """Get all custom (non-default) models"""
        with self.lock:
            # Default models are those defined in _initialize_default_models
            default_model_ids = {
                "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-latest", 
                "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.5-pro"
            }
            
            return [
                model for model_id, model in self.models.items()
                if model_id not in default_model_ids
            ]
    
    def update_model_info(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing model's information"""
        with self.lock:
            if model_id not in self.models:
                return False
            
            model_info = self.models[model_id]
            
            # Update allowed fields
            allowed_fields = {
                'display_name', 'description', 'input_cost_per_1m', 'output_cost_per_1m',
                'context_length', 'supports_streaming', 'supports_function_calling',
                'is_premium', 'cat_personality', 'max_input_tokens', 'max_output_tokens'
            }
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(model_info, field):
                    setattr(model_info, field, value)
            
            # Save configuration and custom models
            self._save_configuration()
            self._save_custom_models()
            
            # If this is a default model being modified, save as model override
            default_model_ids = {
                "gpt-4o", "claude-3-5-sonnet-20241022"
            }
            
            if model_id in default_model_ids:
                self._save_model_overrides()
            
            return True
    
    def _save_custom_models(self):
        """Save custom models to persistent storage using model_config structure"""
        # Get custom models without acquiring lock (since we're already in a locked context)
        default_model_ids = {
            "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-latest", 
            "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash"
        }
        custom_models = [
            model for model_id, model in self.models.items()
            if model_id not in default_model_ids
        ]
        
        # Organize models by provider with sanitized keys
        models_by_provider = {}
        for model in custom_models:
            provider_key = model.provider.value
            if provider_key not in models_by_provider:
                models_by_provider[provider_key] = {}
            
            # Sanitize the model ID for use as Firebase key
            sanitized_key = self.sanitize_key(model.model_id)
            
            model_dict = asdict(model)
            model_dict['provider'] = model.provider.value  # Convert enum to string
            models_by_provider[provider_key][sanitized_key] = model_dict
        
        if self.firebase_db:
            try:
                # Use thread with timeout for Firebase saves
                import threading
                
                def save_model_config():
                    try:
                        # Save each provider's models directly to model_config/{provider}/{model_key}
                        for provider, models in models_by_provider.items():
                            provider_ref = self.firebase_db.child(f'model_config/{provider}')
                            
                            # Get existing provider data (should be a dict of models now)
                            try:
                                existing_models = provider_ref.get()
                                if existing_models is None:
                                    existing_models = {}
                                    print(f"🆕 Creating new model_config/{provider} structure")
                                else:
                                    # Handle migration from old structure with "models" key
                                    if isinstance(existing_models, dict) and 'models' in existing_models:
                                        print(f"🔄 Migrating from old structure with 'models' key")
                                        existing_models = existing_models.get('models', {})
                                    elif isinstance(existing_models, dict):
                                        print(f"📁 Found existing model_config/{provider} data")
                                    else:
                                        print(f"⚠️ Unexpected data type, creating new: {type(existing_models)}")
                                        existing_models = {}
                            except Exception as get_error:
                                print(f"⚠️ Error getting existing data, creating new: {get_error}")
                                existing_models = {}
                            
                            # Merge existing models with new models
                            existing_models.update(models)
                            
                            # Save directly as model_config/{provider} = {model_key: model_data, ...}
                            provider_ref.set(existing_models)
                            print(f"🐱 Saved {len(models)} models to model_config/{provider} (total: {len(existing_models)})")
                        
                        print(f"🎉 Successfully saved model config to Firebase: {len(custom_models)} models across {len(models_by_provider)} providers")
                    except Exception as e:
                        print(f"❌ Failed to save model config in thread: {e}")
                        import traceback
                        traceback.print_exc()
                
                save_thread = threading.Thread(target=save_model_config)
                save_thread.daemon = True
                save_thread.start()
                save_thread.join(timeout=3.0)  # 3 second timeout
                
                if save_thread.is_alive():
                    print("Model config Firebase save timed out")
                    
            except Exception as e:
                print(f"Failed to save model config to Firebase: {e}")
        else:
            try:
                import os
                os.makedirs("data", exist_ok=True)
                with open("data/model_config.json", 'w') as f:
                    json.dump({
                        'model_config': models_by_provider,
                        'last_updated': datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                print(f"Failed to save model config to file: {e}")
    
    def _save_custom_models_sync(self):
        """Save custom models synchronously using model_config structure"""
        print(f"🔄 [SYNC] Starting _save_custom_models_sync...")
        
        try:
            # Get custom models without acquiring lock (since we're already in a locked context)
            default_model_ids = {
                "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-latest", 
                "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.5-pro"
            }
            custom_models = [
                model for model_id, model in self.models.items()
                if model_id not in default_model_ids
            ]
            print(f"📋 [SYNC] Found {len(custom_models)} custom models to save")
            
            # Organize models by provider with sanitized keys
            models_by_provider = {}
            for model in custom_models:
                provider_key = model.provider.value
                if provider_key not in models_by_provider:
                    models_by_provider[provider_key] = {}
                
                # Sanitize the model ID for use as Firebase key
                sanitized_key = self.sanitize_key(model.model_id)
                
                model_dict = asdict(model)
                model_dict['provider'] = model.provider.value  # Convert enum to string
                models_by_provider[provider_key][sanitized_key] = model_dict
                print(f"📝 [SYNC] Prepared model {model.model_id} (key: {sanitized_key}) for provider {provider_key}")
            
            print(f"📊 [SYNC] Models organized by provider: {list(models_by_provider.keys())}")
        except Exception as e:
            print(f"❌ [SYNC] Error preparing model data: {e}")
            import traceback
            traceback.print_exc()
            return
        
        if self.firebase_db:
            print(f"🔥 [SYNC] Firebase is available, attempting save...")
            try:
                import threading
                success = [False]
                error_msg = [None]
                
                def save_model_config():
                    try:
                        print(f"🚀 [SYNC] Starting Firebase save thread...")
                        # Save each provider's models directly to model_config/{provider}/{model_key}
                        for provider, models in models_by_provider.items():
                            print(f"🔥 [SYNC] Processing provider: {provider} with {len(models)} models")
                            provider_ref = self.firebase_db.child(f'model_config/{provider}')
                            
                            # Get existing provider data (should be a dict of models now)
                            try:
                                print(f"📡 [SYNC] Getting existing data from model_config/{provider}...")
                                existing_models = provider_ref.get()
                                if existing_models is None:
                                    existing_models = {}
                                    print(f"🆕 [SYNC] Creating new model_config/{provider} structure")
                                else:
                                    # Handle migration from old structure with "models" key
                                    if isinstance(existing_models, dict) and 'models' in existing_models:
                                        print(f"🔄 [SYNC] Migrating from old structure with 'models' key")
                                        existing_models = existing_models.get('models', {})
                                    elif isinstance(existing_models, dict):
                                        print(f"📁 [SYNC] Found existing model_config/{provider} data: {list(existing_models.keys())}")
                                    else:
                                        print(f"⚠️ [SYNC] Unexpected data type, creating new: {type(existing_models)}")
                                        existing_models = {}
                            except Exception as get_error:
                                print(f"⚠️ [SYNC] Error getting existing data, creating new: {get_error}")
                                existing_models = {}
                            
                            print(f"📋 [SYNC] Existing models in {provider}: {list(existing_models.keys()) if existing_models else 'None'}")
                            
                            # Merge existing models with new models
                            existing_models.update(models)
                            print(f"📋 [SYNC] After merge, {provider} has {len(existing_models)} total models")
                            
                            # Save directly as model_config/{provider} = {model_key: model_data, ...}
                            print(f"💾 [SYNC] Setting data for model_config/{provider}...")
                            provider_ref.set(existing_models)
                            print(f"🐱 [SYNC] Saved {len(models)} models to model_config/{provider} (total: {len(existing_models)})")
                        
                        success[0] = True
                        print(f"🎉 [SYNC] Successfully saved model config to Firebase: {len(custom_models)} models across {len(models_by_provider)} providers")
                    except Exception as e:
                        error_msg[0] = str(e)
                        print(f"❌ [SYNC] Failed to save model config to Firebase: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"🧵 [SYNC] Starting Firebase save thread...")
                # Run with timeout protection
                save_thread = threading.Thread(target=save_model_config)
                save_thread.daemon = True
                save_thread.start()
                save_thread.join(timeout=10.0)  # Increased timeout to 10 seconds
                
                if save_thread.is_alive():
                    print("⏰ [SYNC] Firebase model config save timed out after 10 seconds, falling back to local file")
                elif not success[0]:
                    print(f"❌ [SYNC] Firebase save failed: {error_msg[0]}")
                else:
                    print(f"✅ [SYNC] Firebase save completed successfully")
                    
            except Exception as e:
                print(f"❌ [SYNC] Exception in Firebase save setup: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ [SYNC] Firebase not available, skipping Firebase save")
        
        # Always save to local file as backup
        print(f"💾 [SYNC] Attempting to save to local file as backup...")
        try:
            import os
            os.makedirs("data", exist_ok=True)
            
            # Add metadata to each provider
            providers_with_metadata = {}
            for provider, models in models_by_provider.items():
                providers_with_metadata[provider] = {
                    **models,  # All the models as direct keys
                    '_metadata': {
                        'last_updated': datetime.now().isoformat(),
                        'model_count': len(models)
                    }
                }
            
            with open("data/model_config.json", 'w') as f:
                json.dump({
                    'model_config': providers_with_metadata,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
            print(f"✅ [SYNC] Successfully saved model config to local file")
        except Exception as e:
            print(f"❌ [SYNC] Failed to save model config to file: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"🏁 [SYNC] _save_custom_models_sync completed")
    
    def _save_model_overrides(self):
        """Save modifications to default models using model_config structure"""
        try:
            # Get default model IDs
            default_model_ids = {
                "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro-latest", 
                "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.5-pro"
            }
            
            # Collect overrides for default models and organize by provider
            overrides_by_provider = {}
            for model_id, model_info in self.models.items():
                if model_id in default_model_ids:
                    provider_key = model_info.provider.value
                    if provider_key not in overrides_by_provider:
                        overrides_by_provider[provider_key] = {}
                    
                    # Save the current state as an override
                    override_data = {
                        'model_id': model_info.model_id,
                        'provider': model_info.provider.value,
                        'display_name': model_info.display_name,
                        'description': model_info.description,
                        'input_cost_per_1m': model_info.input_cost_per_1m,
                        'output_cost_per_1m': model_info.output_cost_per_1m,
                        'context_length': model_info.context_length,
                        'max_input_tokens': model_info.max_input_tokens,
                        'max_output_tokens': model_info.max_output_tokens,
                        'supports_streaming': model_info.supports_streaming,
                        'supports_function_calling': model_info.supports_function_calling,
                        'is_premium': model_info.is_premium,
                        'cat_personality': model_info.cat_personality,
                        'is_default_override': True  # Mark as modified default
                    }
                    overrides_by_provider[provider_key][model_id] = override_data
            
            # Save to Firebase using model_config structure
            if self.firebase_db:
                try:
                    import threading
                    
                    def save_overrides():
                        try:
                            for provider, overrides in overrides_by_provider.items():
                                provider_ref = self.firebase_db.child(f'model_config/{provider}')
                                
                                # Get existing provider data
                                existing_data = provider_ref.get() or {}
                                models = existing_data.get('models', {})
                                
                                # Add/update overrides
                                models.update(overrides)
                                
                                # Save updated provider data
                                provider_ref.set({
                                    'models': models,
                                    'last_updated': datetime.now().isoformat()
                                })
                            
                            print(f"🐱 Saved model overrides to model_config: {sum(len(o) for o in overrides_by_provider.values())} models")
                        except Exception as e:
                            print(f"Failed to save model overrides to Firebase: {e}")
                    
                    save_thread = threading.Thread(target=save_overrides)
                    save_thread.daemon = True
                    save_thread.start()
                    save_thread.join(timeout=3.0)
                    
                except Exception as e:
                    print(f"Failed to save model overrides to Firebase: {e}")
            
            # Save to local file as backup
            import os
            os.makedirs("data", exist_ok=True)
            with open("data/model_overrides.json", 'w') as f:
                json.dump({
                    'model_config': overrides_by_provider,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            print(f"Failed to save model overrides: {e}")
    
    def _load_model_overrides(self):
        """Load and apply modifications to default models from model_config structure"""
        try:
            # Try to load overrides from model_config structure in Firebase
            if self.firebase_db:
                try:
                    model_config_ref = self.firebase_db.child('model_config')
                    model_config = model_config_ref.get()
                    
                    if model_config:
                        override_count = 0
                        for provider_key, provider_data in model_config.items():
                            if isinstance(provider_data, dict) and 'models' in provider_data:
                                for model_id, model_data in provider_data['models'].items():
                                    # Check if this is a default model override
                                    if model_data.get('is_default_override', False) and model_id in self.models:
                                        model_info = self.models[model_id]
                                        
                                        # Apply override data
                                        allowed_fields = {
                                            'display_name', 'description', 'input_cost_per_1m', 
                                            'output_cost_per_1m', 'context_length', 'max_input_tokens', 
                                            'max_output_tokens', 'supports_streaming', 
                                            'supports_function_calling', 'is_premium', 'cat_personality'
                                        }
                                        
                                        for field, value in model_data.items():
                                            if field in allowed_fields and hasattr(model_info, field):
                                                setattr(model_info, field, value)
                                        
                                        override_count += 1
                                        print(f"🔧 Applied model_config overrides to {model_id}")
                        
                        if override_count > 0:
                            print(f"🐱 Loaded {override_count} model overrides from model_config")
                            return  # Successfully loaded from new structure
                    
                except Exception as e:
                    print(f"Failed to load model overrides from model_config: {e}")
            
            # Fallback to legacy model_overrides structure
            overrides = {}
            
            # Try to load from Firebase first (legacy)
            if self.firebase_db:
                try:
                    config_ref = self.firebase_db.child('model_overrides')
                    override_data = config_ref.get()
                    if override_data and 'overrides' in override_data:
                        overrides = override_data['overrides']
                        print(f"🐱 Loaded model overrides from legacy Firebase: {len(overrides)} models")
                except Exception as e:
                    print(f"Failed to load legacy model overrides from Firebase: {e}")
            
            # If no Firebase data, try local file
            if not overrides:
                try:
                    if os.path.exists("data/model_overrides.json"):
                        with open("data/model_overrides.json", 'r') as f:
                            override_data = json.load(f)
                        
                        # Check for new model_config structure
                        if 'model_config' in override_data:
                            for provider_key, provider_data in override_data['model_config'].items():
                                for model_id, model_data in provider_data.items():
                                    if model_id in self.models:
                                        overrides[model_id] = model_data
                        # Fallback to legacy structure
                        elif 'overrides' in override_data:
                            overrides = override_data['overrides']
                        
                        if overrides:
                            print(f"🐱 Loaded model overrides from file: {len(overrides)} models")
                except Exception as e:
                    print(f"Failed to load model overrides from file: {e}")
            
            # Apply legacy overrides to default models
            for model_id, override_data in overrides.items():
                if model_id in self.models:
                    model_info = self.models[model_id]
                    for field, value in override_data.items():
                        if hasattr(model_info, field):
                            setattr(model_info, field, value)
                    print(f"🔧 Applied legacy overrides to {model_id}")
                    
        except Exception as e:
            print(f"Failed to load model overrides: {e}")
    
    def _load_custom_models(self):
        """Load custom models from persistent storage using model_config structure"""
        if self.firebase_db:
            try:
                # Use thread with timeout for Firebase loads
                import threading
                loaded_models = []
                
                def load_custom_models():
                    try:
                        # Try new model_config structure first
                        model_config_ref = self.firebase_db.child('model_config')
                        model_config = model_config_ref.get()
                        
                        if model_config:
                            print(f"Loading from model_config structure: {list(model_config.keys()) if model_config else 'None'}")
                            
                            # Load models from each provider
                            for provider_key, provider_data in model_config.items():
                                print(f"Processing provider: {provider_key}")
                                
                                # Handle both old structure (with 'models' key) and new structure (direct models)
                                if isinstance(provider_data, dict):
                                    if 'models' in provider_data:
                                        # Old structure: model_config/{provider}/models/{model_key}
                                        print(f"Using old structure for {provider_key}")
                                        models_dict = provider_data['models']
                                    else:
                                        # New structure: model_config/{provider}/{model_key}
                                        print(f"Using new structure for {provider_key}")
                                        models_dict = provider_data
                                    
                                    for sanitized_key, model_data in models_dict.items():
                                        try:
                                            # Skip non-model entries (like 'last_updated')
                                            if not isinstance(model_data, dict) or 'model_id' not in model_data:
                                                continue
                                                
                                            # Convert provider string back to enum
                                            model_data['provider'] = AIProvider(model_data['provider'])
                                            model_info = ModelInfo(**model_data)
                                            loaded_models.append(model_info)
                                            print(f"Loaded custom model from model_config: {model_info.model_id} (key: {sanitized_key})")
                                        except Exception as model_error:
                                            print(f"Failed to load model {sanitized_key}: {model_error}")
                        else:
                            # Fallback to old custom_models structure for migration
                            print("No model_config found, trying legacy custom_models structure")
                            custom_models_ref = self.firebase_db.child('custom_models')
                            config = custom_models_ref.get()
                            
                            if config and 'custom_models' in config:
                                for model_data in config['custom_models']:
                                    # Convert provider string back to enum
                                    model_data['provider'] = AIProvider(model_data['provider'])
                                    model_info = ModelInfo(**model_data)
                                    loaded_models.append(model_info)
                                    print(f"Loaded custom model from legacy: {model_info.model_id}")
                        
                        print(f"Total loaded models: {len(loaded_models)}")
                    except Exception as e:
                        print(f"Failed to load custom models in thread: {e}")
                
                load_thread = threading.Thread(target=load_custom_models)
                load_thread.daemon = True
                load_thread.start()
                load_thread.join(timeout=5.0)  # 5 second timeout
                
                if load_thread.is_alive():
                    print("Custom models Firebase load timed out")
                else:
                    # Add loaded models to the system
                    for model_info in loaded_models:
                        self.models[model_info.model_id] = model_info
                
            except Exception as e:
                print(f"Failed to load custom models from Firebase: {e}")
        else:
            try:
                # Try new model_config structure first
                if os.path.exists("data/model_config.json"):
                    with open("data/model_config.json", 'r') as f:
                        config = json.load(f)
                    
                    if 'model_config' in config:
                        for provider_key, provider_data in config['model_config'].items():
                            if isinstance(provider_data, dict):
                                for sanitized_key, model_data in provider_data.items():
                                    # Skip metadata and non-model entries
                                    if (sanitized_key.startswith('_') or 
                                        not isinstance(model_data, dict) or 
                                        'model_id' not in model_data):
                                        continue
                                        
                                    # Convert provider string back to enum
                                    model_data['provider'] = AIProvider(model_data['provider'])
                                    model_info = ModelInfo(**model_data)
                                    self.models[model_info.model_id] = model_info
                
                # Fallback to legacy structure
                elif os.path.exists("data/custom_models.json"):
                    with open("data/custom_models.json", 'r') as f:
                        config = json.load(f)
                    
                    if 'custom_models' in config:
                        for model_data in config['custom_models']:
                            # Convert provider string back to enum
                            model_data['provider'] = AIProvider(model_data['provider'])
                            model_info = ModelInfo(**model_data)
                            self.models[model_info.model_id] = model_info
                            
            except Exception as e:
                print(f"Failed to load custom models from file: {e}")
    
    def validate_model_data(self, model_data: Dict[str, Any]) -> Dict[str, str]:
        """Validate model data and return any errors"""
        errors = {}
        
        # Required fields
        required_fields = ['model_id', 'provider', 'display_name', 'description']
        for field in required_fields:
            if not model_data.get(field):
                errors[field] = f"{field} is required"
        
        # Validate provider
        if model_data.get('provider'):
            try:
                AIProvider(model_data['provider'].lower())
            except ValueError:
                errors['provider'] = "Invalid AI provider"
        
        # Validate numeric fields
        numeric_fields = ['input_cost_per_1m', 'output_cost_per_1m', 'context_length']
        for field in numeric_fields:
            if field in model_data:
                try:
                    float(model_data[field])
                    if float(model_data[field]) < 0:
                        errors[field] = f"{field} must be non-negative"
                except (ValueError, TypeError):
                    errors[field] = f"{field} must be a valid number"
        
        # Validate model_id uniqueness
        if model_data.get('model_id') and model_data['model_id'] in self.models:
            errors['model_id'] = "Model ID already exists"
        
        # Validate cat personality
        if model_data.get('cat_personality'):
            valid_personalities = {"curious", "playful", "sleepy", "grumpy", "smart", "fast"}
            if model_data['cat_personality'] not in valid_personalities:
                errors['cat_personality'] = f"Invalid cat personality. Must be one of: {', '.join(valid_personalities)}"
        
        return errors

# Global instance
model_manager = ModelFamilyManager()