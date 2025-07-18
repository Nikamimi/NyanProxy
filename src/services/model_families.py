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
    
    def __init__(self):
        self.lock = threading.Lock()
        self.models: Dict[str, ModelInfo] = {}
        self.whitelisted_models: Dict[AIProvider, Set[str]] = {}
        self.global_usage_stats: Dict[str, ModelUsageStats] = {}
        self.user_usage_stats: Dict[str, Dict[str, ModelUsageStats]] = {}  # user_token -> model_id -> stats
        self.firebase_db = None
        
        # Initialize with default models
        self._initialize_default_models()
        
        # Initialize Firebase if available
        if FIREBASE_AVAILABLE and auth_config.firebase_url:
            self._initialize_firebase()
        
        # Load configuration
        self._load_configuration()
        
        # Load custom models
        self._load_custom_models()
    
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
            ModelInfo(
                model_id="gpt-4o-mini",
                provider=AIProvider.OPENAI,
                display_name="GPT-4o Mini",
                description="Smaller, faster GPT-4o model",
                input_cost_per_1m=0.15,
                output_cost_per_1m=0.6,
                context_length=128000,
                supports_function_calling=True,
                cat_personality="fast"
            ),
            ModelInfo(
                model_id="gpt-3.5-turbo",
                provider=AIProvider.OPENAI,
                display_name="GPT-3.5 Turbo",
                description="Fast and efficient model for most tasks",
                input_cost_per_1m=0.5,
                output_cost_per_1m=1.5,
                context_length=16385,
                supports_function_calling=True,
                cat_personality="playful"
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
            ModelInfo(
                model_id="claude-3-haiku-20240307",
                provider=AIProvider.ANTHROPIC,
                display_name="Claude 3 Haiku",
                description="Fastest Claude model for simple tasks",
                input_cost_per_1m=0.25,
                output_cost_per_1m=1.25,
                context_length=200000,
                cat_personality="fast"
            ),
            
            # Google Models (prices per 1M tokens)
            ModelInfo(
                model_id="gemini-1.5-pro",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 1.5 Pro",
                description="Google's most capable model",
                input_cost_per_1m=3.5,
                output_cost_per_1m=10.5,
                context_length=2000000,
                is_premium=True,
                cat_personality="curious"
            ),
            ModelInfo(
                model_id="gemini-1.5-flash",
                provider=AIProvider.GOOGLE,
                display_name="Gemini 1.5 Flash",
                description="Fast and efficient Gemini model",
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.3,
                context_length=1000000,
                cat_personality="fast"
            ),
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
        
        # Initialize default whitelists (all models enabled by default)
        for provider in AIProvider:
            self.whitelisted_models[provider] = set()
            provider_models = [m.model_id for m in default_models if m.provider == provider]
            self.whitelisted_models[provider].update(provider_models)
    
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
                        self.whitelisted_models[provider] = set(model_list)
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
                                self.whitelisted_models[provider] = set(model_list)
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
        config = {
            'whitelisted_models': {
                provider.value: list(models) 
                for provider, models in self.whitelisted_models.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
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
        config = {
            'whitelisted_models': {
                provider.value: list(models) 
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
            return model_id in self.whitelisted_models.get(provider, set())
    
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
                return False
            
            if provider not in self.whitelisted_models:
                self.whitelisted_models[provider] = set()
            
            self.whitelisted_models[provider].add(model_id)
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
            # Validate all model IDs exist
            valid_ids = [mid for mid in model_ids if mid in self.models]
            
            self.whitelisted_models[provider] = set(valid_ids)
            self._save_configuration()
            return len(valid_ids) == len(model_ids)
    
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
            
            return cost
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        with self.lock:
            return self.models.get(model_id)
    
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
                return False
            
            # Add the model to in-memory storage first
            self.models[model_info.model_id] = model_info
            
            # Auto-whitelist if requested
            if auto_whitelist:
                provider = model_info.provider
                if provider not in self.whitelisted_models:
                    self.whitelisted_models[provider] = set()
                self.whitelisted_models[provider].add(model_info.model_id)
            
            print(f"Successfully added custom model: {model_info.model_id} to provider {model_info.provider.value}")
            
            # Schedule async save in background (non-blocking)
            import threading
            def background_save():
                self._save_configuration()
                self._save_custom_models()
            
            save_thread = threading.Thread(target=background_save)
            save_thread.daemon = True
            save_thread.start()
            
            return True
    
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
                "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
                "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307",
                "gemini-1.5-pro", "gemini-1.5-flash"
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
                'is_premium', 'cat_personality'
            }
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(model_info, field):
                    setattr(model_info, field, value)
            
            # Save configuration
            self._save_configuration()
            self._save_custom_models()
            
            return True
    
    def _save_custom_models(self):
        """Save custom models to persistent storage"""
        custom_models = self.get_custom_models()
        
        # Convert to serializable format
        models_data = []
        for model in custom_models:
            model_dict = asdict(model)
            model_dict['provider'] = model.provider.value  # Convert enum to string
            models_data.append(model_dict)
        
        config = {
            'custom_models': models_data,
            'last_updated': datetime.now().isoformat()
        }
        
        if self.firebase_db:
            try:
                # Use thread with timeout for Firebase saves
                import threading
                
                def save_custom_models():
                    try:
                        custom_models_ref = self.firebase_db.child('custom_models')
                        custom_models_ref.set(config)
                        print(f"Saved custom models to Firebase: {len(config['custom_models'])} models")
                        print(f"Config structure: {config}")
                    except Exception as e:
                        print(f"Failed to save custom models in thread: {e}")
                
                save_thread = threading.Thread(target=save_custom_models)
                save_thread.daemon = True
                save_thread.start()
                save_thread.join(timeout=3.0)  # 3 second timeout
                
                if save_thread.is_alive():
                    print("Custom models Firebase save timed out")
                    
            except Exception as e:
                print(f"Failed to save custom models to Firebase: {e}")
        else:
            try:
                import os
                os.makedirs("data", exist_ok=True)
                with open("data/custom_models.json", 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                print(f"Failed to save custom models to file: {e}")
    
    def _save_custom_models_sync(self):
        """Save custom models synchronously (for critical operations like model addition)"""
        custom_models = self.get_custom_models()
        
        # Convert to serializable format
        models_data = []
        for model in custom_models:
            model_dict = asdict(model)
            model_dict['provider'] = model.provider.value  # Convert enum to string
            models_data.append(model_dict)
        
        config = {
            'custom_models': models_data,
            'last_updated': datetime.now().isoformat()
        }
        
        if self.firebase_db:
            try:
                import threading
                success = [False]
                
                def save_custom_models():
                    try:
                        custom_models_ref = self.firebase_db.child('custom_models')
                        custom_models_ref.set(config)
                        success[0] = True
                        print("Saved custom models to Firebase (sync)")
                    except Exception as e:
                        print(f"Failed to save custom models to Firebase (sync): {e}")
                
                # Run with timeout protection
                save_thread = threading.Thread(target=save_custom_models)
                save_thread.daemon = True
                save_thread.start()
                save_thread.join(timeout=5.0)  # 5 second timeout
                
                if save_thread.is_alive():
                    print("Firebase custom models save timed out, falling back to local file")
                    
            except Exception as e:
                print(f"Failed to save custom models to Firebase (sync): {e}")
        
        # Always save to local file as backup
        try:
            import os
            os.makedirs("data", exist_ok=True)
            with open("data/custom_models.json", 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save custom models to file: {e}")
    
    def _load_custom_models(self):
        """Load custom models from persistent storage"""
        if self.firebase_db:
            try:
                # Use thread with timeout for Firebase loads
                import threading
                loaded_models = []
                
                def load_custom_models():
                    try:
                        custom_models_ref = self.firebase_db.child('custom_models')
                        config = custom_models_ref.get()
                        
                        print(f"Firebase custom models config: {config}")
                        
                        if config and 'custom_models' in config:
                            for model_data in config['custom_models']:
                                # Convert provider string back to enum
                                model_data['provider'] = AIProvider(model_data['provider'])
                                model_info = ModelInfo(**model_data)
                                loaded_models.append(model_info)
                                print(f"Loaded custom model: {model_info.model_id}")
                        else:
                            print("No custom_models found in Firebase config")
                        print(f"Loaded {len(loaded_models)} custom models from Firebase")
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
                if os.path.exists("data/custom_models.json"):
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