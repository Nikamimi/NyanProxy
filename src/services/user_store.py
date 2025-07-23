import uuid
import hashlib
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: Firebase not available. User data will be stored in memory only.")

from ..config.auth import auth_config

class ReadWriteLock:
    """A reader-writer lock implementation for better concurrency"""
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        self._writers = 0
        self._write_ready = threading.Condition(threading.RLock())
        
    @contextmanager
    def read_lock(self):
        """Acquire read lock"""
        with self._read_ready:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
        
        try:
            yield
        finally:
            with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()
    
    @contextmanager
    def write_lock(self):
        """Acquire write lock"""
        with self._write_ready:
            while self._writers > 0:
                self._write_ready.wait()
            self._writers += 1
        
        with self._read_ready:
            while self._readers > 0:
                self._read_ready.wait()
        
        try:
            yield
        finally:
            with self._write_ready:
                self._writers -= 1
                self._write_ready.notify_all()
            with self._read_ready:
                self._read_ready.notify_all()

class UserType(Enum):
    NORMAL = "normal"
    SPECIAL = "special"
    TEMPORARY = "temporary"

class UserStatus(Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    SUSPENDED = "suspended"
    EXPIRED = "expired"

class AuthResult(Enum):
    SUCCESS = "success"
    DISABLED = "disabled"
    NOT_FOUND = "not_found"
    LIMITED = "limited"

@dataclass
class TokenCount:
    input: int = 0
    output: int = 0
    total: int = 0
    requests: int = 0
    cost_usd: float = 0.0
    first_request: Optional[datetime] = None
    last_request: Optional[datetime] = None
    
    def add_usage(self, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Add usage data and update counts"""
        self.input += input_tokens
        self.output += output_tokens
        self.total += input_tokens + output_tokens
        self.requests += 1
        self.cost_usd += cost
        
        now = datetime.now()
        if self.first_request is None:
            self.first_request = now
        self.last_request = now
    
    def to_dict(self):
        return {
            'input': self.input,
            'output': self.output,
            'total': self.total,
            'requests': self.requests,
            'cost_usd': self.cost_usd,
            'first_request': self.first_request.isoformat() if self.first_request else None,
            'last_request': self.last_request.isoformat() if self.last_request else None
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            input=data.get('input', 0),
            output=data.get('output', 0),
            total=data.get('total', 0),
            requests=data.get('requests', 0),
            cost_usd=data.get('cost_usd', 0.0),
            first_request=datetime.fromisoformat(data['first_request']) if data.get('first_request') else None,
            last_request=datetime.fromisoformat(data['last_request']) if data.get('last_request') else None
        )

@dataclass
class IPUsage:
    ip: str
    prompt_count: int = 0
    first_seen: Optional[datetime] = None
    last_used: Optional[datetime] = None
    total_requests: int = 0
    total_tokens: int = 0
    models_used: Dict[str, int] = None
    user_agent: Optional[str] = None
    is_suspicious: bool = False
    
    def __post_init__(self):
        if self.models_used is None:
            self.models_used = {}
        if self.first_seen is None:
            self.first_seen = datetime.now()
    
    def add_request(self, model_family: str, tokens: int, user_agent: str = None):
        """Add request data and update counts"""
        self.prompt_count += 1
        self.total_requests += 1
        self.total_tokens += tokens
        self.last_used = datetime.now()
        
        if user_agent:
            self.user_agent = user_agent
        
        # Track model usage
        if model_family not in self.models_used:
            self.models_used[model_family] = 0
        self.models_used[model_family] += 1
    
    def to_dict(self):
        return {
            'ip': self.ip,
            'prompt_count': self.prompt_count,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'models_used': {k.replace('.', '_DOT_').replace('#', '_HASH_').replace('$', '_DOLLAR_').replace('[', '_LBRACKET_').replace(']', '_RBRACKET_').replace('/', '_SLASH_'): v for k, v in self.models_used.items()},
            'user_agent': self.user_agent,
            'is_suspicious': self.is_suspicious
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            ip=data['ip'],
            prompt_count=data.get('prompt_count', 0),
            first_seen=datetime.fromisoformat(data['first_seen']) if data.get('first_seen') else None,
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            total_requests=data.get('total_requests', 0),
            total_tokens=data.get('total_tokens', 0),
            models_used={k.replace('_DOT_', '.').replace('_HASH_', '#').replace('_DOLLAR_', '$').replace('_LBRACKET_', '[').replace('_RBRACKET_', ']').replace('_SLASH_', '/'): v for k, v in data.get('models_used', {}).items()},
            user_agent=data.get('user_agent'),
            is_suspicious=data.get('is_suspicious', False)
        )

@dataclass
class User:
    token: str
    type: UserType = UserType.NORMAL
    created_at: datetime = None
    last_used: Optional[datetime] = None
    disabled_at: Optional[datetime] = None
    disabled_reason: Optional[str] = None
    status: UserStatus = UserStatus.ACTIVE
    ip: List[str] = None
    ip_usage: List[IPUsage] = None
    token_counts: Dict[str, TokenCount] = None
    prompt_limits: Dict[str, Optional[int]] = None
    token_refresh: Dict[str, int] = None  # Token refresh timestamps per model family
    nickname: Optional[str] = None
    
    # Temporary user limits
    prompt_limits: Optional[int] = None  # Maximum number of prompts for temporary users
    max_ips: Optional[int] = None  # Maximum number of IPs for this user
    
    # Enhanced tracking features
    total_requests: int = 0
    total_cost: float = 0.0
    rate_limit_hits: int = 0
    last_rate_limit: Optional[datetime] = None
    suspicious_activity_count: int = 0
    preferred_models: Dict[str, int] = None
    average_tokens_per_request: float = 0.0
    peak_requests_per_hour: int = 0
    favorite_user_agent: Optional[str] = None
    
    # Cat-themed status tracking
    mood: str = "happy"  # happy, sleepy, grumpy, playful
    favorite_treat: Optional[str] = None
    paw_prints: int = 0  # number of successful requests
    
    # Happiness/Merit System
    happiness: int = 100  # 0-100 scale
    suspicious_events: int = 0  # counter for suspicious activities
    token_violations: int = 0  # token limit violations
    rate_limit_violations: int = 0  # rate limit violations
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.ip is None:
            self.ip = []
        if self.ip_usage is None:
            self.ip_usage = []
        if self.token_counts is None:
            self.token_counts = {}
        if self.prompt_limits is None:
            # Default request limits per model family (None = unlimited)
            self.prompt_limits = {
                'openai': None,
                'anthropic': None,
                'google': None,
                'mistral': None
            }
        if self.token_refresh is None:
            # Initialize token refresh timestamps (0 = never refreshed)
            self.token_refresh = {
                'openai': 0,
                'anthropic': 0,
                'google': 0,
                'mistral': 0
            }
        if self.preferred_models is None:
            self.preferred_models = {}
        
        # Initialize happiness/merit system fields for backward compatibility
        if not hasattr(self, 'happiness'):
            self.happiness = 100
        if not hasattr(self, 'suspicious_events'):
            self.suspicious_events = 0
        if not hasattr(self, 'token_violations'):
            self.token_violations = 0
        if not hasattr(self, 'rate_limit_violations'):
            self.rate_limit_violations = 0
        
        # CRITICAL: Ensure status field is always properly set
        if not hasattr(self, 'status') or self.status is None:
            # Determine status based on disabled_at if status is missing
            if self.disabled_at:
                self.status = UserStatus.DISABLED
                print(f"[RETRY] POST_INIT: Set user {self.token[:8]} status to DISABLED based on disabled_at")
            else:
                self.status = UserStatus.ACTIVE
                print(f"[RETRY] POST_INIT: Set user {self.token[:8]} status to ACTIVE (no disabled_at)")
            
        # Initialize cat mood based on user type
        if self.type == UserType.SPECIAL:
            self.mood = "playful"
        elif self.type == UserType.TEMPORARY:
            self.mood = "sleepy"
        else:
            self.mood = "happy"
    
    def is_disabled(self) -> bool:
        # Primary check: use explicit status field if available
        if hasattr(self, 'status') and self.status is not None:
            return self.status in [UserStatus.DISABLED, UserStatus.SUSPENDED, UserStatus.EXPIRED]
        
        # Fallback: use legacy disabled_at field for backward compatibility
        return self.disabled_at is not None
    
    def is_ip_limit_exceeded(self) -> bool:
        # Use user-specific max_ips if set, otherwise use global setting
        max_ips = self.max_ips if self.max_ips is not None else auth_config.max_ips_per_user
        return len(self.ip) >= max_ips
    
    def is_prompt_limit_exceeded(self) -> bool:
        """Check if user has exceeded their prompt limits (for temporary users)"""
        if self.prompt_limits is None:
            return False
        
        total_prompts = sum(ip_usage.prompt_count for ip_usage in self.ip_usage)
        return total_prompts >= self.prompt_limits
    
    def get_remaining_prompts(self) -> Optional[int]:
        """Get remaining prompts across all model families (for backward compatibility)"""
        if not self.prompt_limits:
            return None
        
        # Find the most restrictive limit (minimum non-null value)
        limits = [limit for limit in self.prompt_limits.values() if limit is not None]
        if not limits:
            return None  # All unlimited
        
        min_limit = min(limits)
        total_requests = sum(count.requests for count in self.token_counts.values())
        return max(0, min_limit - total_requests)
    
    def get_token_usage(self, model_family: str) -> TokenCount:
        """Get aggregated token usage for a model family"""
        # If we have direct family tracking, use it
        if model_family in self.token_counts:
            return self.token_counts[model_family]
        
        # Otherwise, aggregate usage from all models in this family
        aggregated = TokenCount()
        
        # Define which models belong to which families
        family_patterns = {
            'openai': ['gpt', 'o1', 'o3'],
            'anthropic': ['claude'],
            'google': ['gemini'],
            'mistral': ['mistral'],
            'groq': ['groq'],
            'cohere': ['cohere']
        }
        
        patterns = family_patterns.get(model_family, [])
        
        for model_name, token_count in self.token_counts.items():
            # Check if this model belongs to the requested family
            model_lower = model_name.lower()
            if any(pattern in model_lower for pattern in patterns):
                aggregated.input += token_count.input
                aggregated.output += token_count.output
                aggregated.total += token_count.total
                aggregated.requests += token_count.requests
                aggregated.cost_usd += token_count.cost_usd
                
                # Update first/last request times
                if token_count.first_request:
                    if not aggregated.first_request or token_count.first_request < aggregated.first_request:
                        aggregated.first_request = token_count.first_request
                if token_count.last_request:
                    if not aggregated.last_request or token_count.last_request > aggregated.last_request:
                        aggregated.last_request = token_count.last_request
        
        return aggregated
    
    def get_request_limit(self, model_family: str) -> Optional[int]:
        """Get request limit for model family. Returns None for unlimited."""
        return self.prompt_limits.get(model_family, None)
    
    def get_token_limit(self, model_family: str) -> Optional[int]:
        """Legacy method for backward compatibility"""
        return self.get_request_limit(model_family)
    
    def get_token_refresh(self, model_family: str) -> int:
        """Get token refresh timestamp for model family"""
        return self.token_refresh.get(model_family, 0)
    
    def refresh_tokens(self, model_family: str = None) -> bool:
        """Refresh token counts for specified model family or all families"""
        import time
        current_timestamp = int(time.time())
        
        if model_family:
            # Refresh specific model family
            if model_family in self.token_counts:
                self.token_counts[model_family] = TokenCount()
            self.token_refresh[model_family] = current_timestamp
            return True
        else:
            # Refresh all model families
            for family in self.prompt_limits.keys():
                if family in self.token_counts:
                    self.token_counts[family] = TokenCount()
                self.token_refresh[family] = current_timestamp
            return True
    
    def add_request_tracking(self, model_family: str, input_tokens: int, output_tokens: int, 
                           cost: float, ip_hash: str, user_agent: str = None):
        """Add comprehensive request tracking with cat-themed elements"""
        # Update token counts
        if model_family not in self.token_counts:
            self.token_counts[model_family] = TokenCount()
        self.token_counts[model_family].add_usage(input_tokens, output_tokens, cost)
        
        # Update general stats
        self.total_requests += 1
        self.total_cost += cost
        self.last_used = datetime.now()
        self.paw_prints += 1  # Successful request = paw print!
        
        # Update IP usage tracking
        ip_usage = self.get_or_create_ip_usage(ip_hash)
        ip_usage.add_request(model_family, input_tokens + output_tokens, user_agent)
        
        # Update preferred models
        if model_family not in self.preferred_models:
            self.preferred_models[model_family] = 0
        self.preferred_models[model_family] += 1
        
        # Update average tokens per request
        total_tokens = sum(count.total for count in self.token_counts.values())
        self.average_tokens_per_request = total_tokens / self.total_requests if self.total_requests > 0 else 0
        
        # Update mood based on usage patterns
        self.update_mood()
        
        # Update favorite treat based on most used model (now showing actual model names)
        most_used_model = max(self.preferred_models.items(), key=lambda x: x[1], default=('', 0))[0]
        if most_used_model:
            self.favorite_treat = most_used_model
        else:
            self.favorite_treat = 'Still exploring treats... '
    
    def get_or_create_ip_usage(self, ip_hash: str) -> IPUsage:
        """Get or create IP usage tracking for a given IP hash"""
        for ip_usage in self.ip_usage:
            if ip_usage.ip == ip_hash:
                return ip_usage
        
        # Create new IP usage if not found
        new_ip_usage = IPUsage(ip=ip_hash)
        self.ip_usage.append(new_ip_usage)
        
        # Also add to IP list if not already there
        if ip_hash not in self.ip:
            self.ip.append(ip_hash)
            
        return new_ip_usage
    
    def update_mood(self):
        """Update cat mood based on usage patterns"""
        if self.is_disabled():
            self.mood = "grumpy"
        elif self.rate_limit_hits > 10:
            self.mood = "grumpy"
        elif self.total_requests > 1000:
            self.mood = "playful"
        elif self.type == UserType.TEMPORARY:
            self.mood = "sleepy"
        else:
            self.mood = "happy"
    
    def get_mood_emoji(self) -> str:
        """Get emoji representation of cat mood"""
        mood_emojis = {
            "happy": "",
            "sleepy": "",
            "grumpy": "",
            "playful": ""
        }
        return mood_emojis.get(self.mood, "")
    
    def validate_and_fix_data_integrity(self) -> bool:
        """Validate and fix data integrity issues with user object"""
        fixed_issues = False
        
        try:
            # Ensure token exists
            if not hasattr(self, 'token') or not self.token:
                print(f"[ERROR] INTEGRITY: User missing token field")
                return False  # Can't fix missing token
            
            # Ensure type exists and is valid
            if not hasattr(self, 'type') or self.type is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing type, setting to NORMAL")
                self.type = UserType.NORMAL
                fixed_issues = True
            
            # Ensure created_at exists
            if not hasattr(self, 'created_at') or self.created_at is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing created_at, setting to now")
                self.created_at = datetime.now()
                fixed_issues = True
            
            # Ensure basic collections exist
            if not hasattr(self, 'ip') or self.ip is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing ip list, initializing")
                self.ip = []
                fixed_issues = True
            
            if not hasattr(self, 'ip_usage') or self.ip_usage is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing ip_usage list, initializing")
                self.ip_usage = []
                fixed_issues = True
            
            if not hasattr(self, 'token_counts') or self.token_counts is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing token_counts, initializing")
                self.token_counts = {}
                fixed_issues = True
            
            if not hasattr(self, 'prompt_limits') or self.prompt_limits is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing prompt_limits, initializing")
                self.prompt_limits = {
                    'openai': None,
                    'anthropic': None,
                    'google': None,
                    'mistral': None
                }
                fixed_issues = True
            
            if not hasattr(self, 'token_refresh') or self.token_refresh is None:
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing token_refresh, initializing")
                self.token_refresh = {
                    'openai': 0,
                    'anthropic': 0,
                    'google': 0,
                    'mistral': 0
                }
                fixed_issues = True
            
            # Ensure optional fields have safe defaults
            if not hasattr(self, 'nickname'):
                self.nickname = None
            
            if not hasattr(self, 'disabled_at'):
                self.disabled_at = None
            
            if not hasattr(self, 'disabled_reason'):
                self.disabled_reason = None
            
            if not hasattr(self, 'last_used'):
                self.last_used = None
            
            # Ensure numeric fields have safe defaults
            numeric_fields = [
                ('total_requests', 0),
                ('total_cost', 0.0),
                ('rate_limit_hits', 0),
                ('suspicious_activity_count', 0),
                ('average_tokens_per_request', 0.0),
                ('peak_requests_per_hour', 0),
                ('paw_prints', 0)
            ]
            
            for field_name, default_value in numeric_fields:
                if not hasattr(self, field_name):
                    setattr(self, field_name, default_value)
                    if field_name in ['paw_prints', 'mood']:  # Only log important cat fields
                        print(f"[TOOL] INTEGRITY: User {self.token[:8]} missing {field_name}, set to {default_value}")
                        fixed_issues = True
            
            # Ensure cat-themed fields exist
            if not hasattr(self, 'mood') or not self.mood:
                self.mood = 'happy'
                fixed_issues = True
            
            if not hasattr(self, 'favorite_treat'):
                self.favorite_treat = None
            
            # Ensure collections are not corrupted
            if not isinstance(self.preferred_models, dict):
                print(f"[TOOL] INTEGRITY: User {self.token[:8]} has corrupted preferred_models, resetting")
                self.preferred_models = {}
                fixed_issues = True
            
            return fixed_issues
            
        except Exception as e:
            print(f"[ERROR] INTEGRITY: Error validating user {getattr(self, 'token', 'unknown')[:8]}: {e}")
            return False
    
    def add_rate_limit_hit(self):
        """Track rate limit violations"""
        self.rate_limit_hits += 1
        self.last_rate_limit = datetime.now()
        self.update_mood()
    
    def mark_suspicious_activity(self):
        """Mark suspicious activity for this user"""
        self.suspicious_activity_count += 1
        if self.suspicious_activity_count > 5:
            self.mood = "grumpy"
    
    def record_token_violation(self):
        """Record a token limit violation and decrease happiness"""
        self.token_violations += 1
        self.suspicious_events += 1
        self.happiness = max(0, self.happiness - 10)
        self._update_happiness_mood()
    
    def record_rate_limit_violation(self):
        """Record a rate limit violation and decrease happiness"""
        self.rate_limit_violations += 1
        self.suspicious_events += 1
        self.happiness = max(0, self.happiness - 10)
        self._update_happiness_mood()
    
    def record_successful_prompt(self):
        """Record a successful prompt and increase happiness slightly"""
        self.happiness = min(100, self.happiness + 1)
        self._update_happiness_mood()
    
    def _update_happiness_mood(self):
        """Update mood based on happiness level"""
        if self.happiness >= 80:
            self.mood = "happy"
        elif self.happiness >= 60:
            self.mood = "playful"
        elif self.happiness >= 40:
            self.mood = "sleepy"
        else:
            self.mood = "grumpy"
    
    def get_happiness_percentage(self) -> int:
        """Get happiness as a percentage (0-100)"""
        return self.happiness
    
    def get_happiness_emoji(self) -> str:
        """Get emoji representing current happiness level"""
        if self.happiness >= 80:
            return ""  # grinning cat
        elif self.happiness >= 60:
            return ""  # smiling cat
        elif self.happiness >= 40:
            return ""  # cat with wry smile
        else:
            return ""  # weary cat
    
    def _sanitize_model_key(self, key: str) -> str:
        """Sanitize model name for Firebase (replace invalid characters)"""
        return key.replace('.', '_DOT_').replace('#', '_HASH_').replace('$', '_DOLLAR_').replace('[', '_LBRACKET_').replace(']', '_RBRACKET_').replace('/', '_SLASH_')
    
    def _unsanitize_model_key(self, key: str) -> str:
        """Reverse sanitization of model name from Firebase"""
        return key.replace('_DOT_', '.').replace('_HASH_', '#').replace('_DOLLAR_', '$').replace('_LBRACKET_', '[').replace('_RBRACKET_', ']').replace('_SLASH_', '/')
    
    def to_dict(self) -> dict:
        return {
            'token': self.token,
            'type': self.type.value,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'disabled_at': self.disabled_at.isoformat() if self.disabled_at else None,
            'disabled_reason': self.disabled_reason,
            'status': self.status.value if hasattr(self, 'status') and self.status else ('disabled' if self.disabled_at else 'active'),
            'ip': self.ip,
            'ip_usage': [usage.to_dict() for usage in self.ip_usage],
            'token_counts': {self._sanitize_model_key(k): v.to_dict() for k, v in self.token_counts.items()},
            'token_refresh': self.token_refresh,
            'nickname': self.nickname,
            'prompt_limits': self.prompt_limits,
            'max_ips': self.max_ips,
            
            # Enhanced tracking fields
            'total_requests': self.total_requests,
            'total_cost': self.total_cost,
            'rate_limit_hits': self.rate_limit_hits,
            'last_rate_limit': self.last_rate_limit.isoformat() if self.last_rate_limit else None,
            'suspicious_activity_count': self.suspicious_activity_count,
            'preferred_models': {self._sanitize_model_key(k): v for k, v in self.preferred_models.items()},
            'average_tokens_per_request': self.average_tokens_per_request,
            'peak_requests_per_hour': self.peak_requests_per_hour,
            'favorite_user_agent': self.favorite_user_agent,
            
            # Cat-themed fields
            'mood': self.mood,
            'mood_emoji': self.get_mood_emoji(),
            'favorite_treat': self.favorite_treat,
            'paw_prints': self.paw_prints,
            
            # Happiness/Merit System
            'happiness': self.happiness,
            'suspicious_events': self.suspicious_events,
            'token_violations': self.token_violations,
            'rate_limit_violations': self.rate_limit_violations
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        # Handle backward compatibility for created_at
        created_at = data.get('created_at')
        if created_at:
            if isinstance(created_at, (int, float)):
                # Legacy timestamp format (milliseconds)
                created_at = datetime.fromtimestamp(created_at / 1000)
            else:
                # New ISO format
                created_at = datetime.fromisoformat(created_at)
        elif data.get('createdAt'):
            # Legacy field name with timestamp
            created_at = datetime.fromtimestamp(data['createdAt'] / 1000)
        else:
            # Use current time for legacy users without created_at
            created_at = datetime.now()
            
        # Handle legacy lastUsedAt timestamp
        last_used = data.get('last_used')
        if last_used:
            if isinstance(last_used, (int, float)):
                last_used = datetime.fromtimestamp(last_used / 1000)
            else:
                last_used = datetime.fromisoformat(last_used)
        elif data.get('lastUsedAt'):
            last_used = datetime.fromtimestamp(data['lastUsedAt'] / 1000)
        else:
            last_used = None
        
        # Handle backward compatibility for token_counts
        token_counts = {}
        raw_token_counts = data.get('token_counts', {})
        
        # Check if this is old schema (flat numbers) or new schema (dict objects)
        for k, v in raw_token_counts.items():
            # Unsanitize the key (convert Firebase-safe key back to original model name)
            original_key = k.replace('_DOT_', '.').replace('_HASH_', '#').replace('_DOLLAR_', '$').replace('_LBRACKET_', '[').replace('_RBRACKET_', ']').replace('_SLASH_', '/')
            
            if isinstance(v, dict):
                if 'input' in v and 'output' in v:
                    # New format with detailed token tracking
                    token_counts[original_key] = TokenCount.from_dict(v)
                else:
                    # Old format without detailed tracking
                    token_counts[original_key] = TokenCount(
                        total=v.get('total', 0),
                        requests=v.get('requests', 0),
                        cost_usd=v.get('cost_usd', 0.0)
                    )
            elif isinstance(v, (int, float)):
                # Very old format - just a number representing total tokens
                token_counts[original_key] = TokenCount(total=int(v))
        
        # Handle backward compatibility for legacy promptCount and ipUsage
        if 'promptCount' in data and 'ip_usage' not in data:
            # Convert old promptCount to new ip_usage format
            ip_usage = []
            if data.get('ipUsage'):
                for old_ip in data['ipUsage']:
                    ip_usage.append(IPUsage(
                        ip=old_ip.get('ip', ''),
                        prompt_count=old_ip.get('prompt', 0),
                        total_requests=old_ip.get('prompt', 0),
                        last_used=datetime.fromtimestamp(old_ip['lastUsedAt']/1000) if old_ip.get('lastUsedAt') else None
                    ))
        else:
            ip_usage = [IPUsage.from_dict(usage) for usage in data.get('ip_usage', [])]
        
        return cls(
            token=data['token'],
            type=UserType(data.get('type', 'normal')),
            created_at=created_at,
            last_used=last_used,
            disabled_at=datetime.fromisoformat(data['disabled_at']) if data.get('disabled_at') else None,
            disabled_reason=data.get('disabled_reason'),
            status=UserStatus(data.get('status', 'active')) if data.get('status') else (UserStatus.DISABLED if data.get('disabled_at') else UserStatus.ACTIVE),
            ip=data.get('ip', []),
            ip_usage=ip_usage,
            token_counts=token_counts,
            prompt_limits=data.get('prompt_limits', data.get('token_limits', data.get('tokenLimits', {}))),
            token_refresh=data.get('token_refresh', data.get('tokenRefresh', {})),
            nickname=data.get('nickname'),
            max_ips=data.get('max_ips'),
            
            # Enhanced tracking fields with backward compatibility
            total_requests=data.get('total_requests', data.get('promptCount', 0)),
            total_cost=data.get('total_cost', 0.0),
            rate_limit_hits=data.get('rate_limit_hits', 0),
            last_rate_limit=datetime.fromisoformat(data['last_rate_limit']) if data.get('last_rate_limit') else None,
            suspicious_activity_count=data.get('suspicious_activity_count', 0),
            preferred_models={k.replace('_DOT_', '.').replace('_HASH_', '#').replace('_DOLLAR_', '$').replace('_LBRACKET_', '[').replace('_RBRACKET_', ']').replace('_SLASH_', '/'): v for k, v in data.get('preferred_models', {}).items()},
            average_tokens_per_request=data.get('average_tokens_per_request', 0.0),
            peak_requests_per_hour=data.get('peak_requests_per_hour', 0),
            favorite_user_agent=data.get('favorite_user_agent'),
            
            # Cat-themed fields
            mood=data.get('mood', 'happy'),
            favorite_treat=data.get('favorite_treat'),
            paw_prints=data.get('paw_prints', 0),
            
            # Happiness/Merit System
            happiness=data.get('happiness', 100),
            suspicious_events=data.get('suspicious_events', 0),
            token_violations=data.get('token_violations', 0),
            rate_limit_violations=data.get('rate_limit_violations', 0)
        )

class UserStore:
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.lock = ReadWriteLock()  # Use reader-writer lock for better concurrency
        self.firebase_db = None
        self.flush_queue = set()
        self.flush_queue_lock = threading.Lock()  # Separate lock for flush queue
        self.cleanup_thread = None
        
        # Initialize Firebase if available
        if FIREBASE_AVAILABLE and auth_config.firebase_url:
            self._initialize_firebase()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Initializing Firebase
            
            if auth_config.firebase_service_account_key:
                # Use service account key
                # Using service account key
                key_str = auth_config.firebase_service_account_key
                decoded_key = json.loads(key_str)
                cred = credentials.Certificate(decoded_key)
            else:
                # Use default credentials
                # Using default credentials
                cred = credentials.ApplicationDefault()
            
            if not firebase_admin._apps:
                # Initializing Firebase app
                firebase_admin.initialize_app(cred, {
                    'databaseURL': auth_config.firebase_url
                })
            
            self.firebase_db = db.reference()
            # Firebase initialized successfully
            
            # Load existing users from Firebase
            self._load_users_from_firebase()
            
        except Exception as e:
            print(f"Failed to initialize Firebase: {e}")
            self.firebase_db = None
    
    def _load_users_from_firebase(self):
        """Load existing users from Firebase"""
        try:
            if not self.firebase_db:
                return
            
            users_ref = self.firebase_db.child('users')
            users_data = users_ref.get()
            
            if users_data:
                with self.lock.write_lock():
                    users_to_migrate = []
                    for token, user_data in users_data.items():
                        try:
                            print(f"[RETRY] LOADING: User {token[:8]} from Firebase")
                            print(f"[RETRY] LOADING: User {token[:8]} Firebase data has status: {'status' in user_data}")
                            print(f"[RETRY] LOADING: User {token[:8]} Firebase status value: {user_data.get('status')}")
                            print(f"[RETRY] LOADING: User {token[:8]} Firebase disabled_at: {user_data.get('disabled_at')}")
                            
                            user = User.from_dict(user_data)
                            
                            print(f"[RETRY] LOADING: User {token[:8]} loaded - status: {getattr(user, 'status', 'not set')}")
                            print(f"[RETRY] LOADING: User {token[:8]} loaded - hasattr status: {hasattr(user, 'status')}")
                            print(f"[RETRY] LOADING: User {token[:8]} loaded - is_disabled(): {user.is_disabled()}")
                            print(f"[RETRY] LOADING: User {token[:8]} token_counts keys: {list(user.token_counts.keys()) if user.token_counts else 'None'}")
                            if user.token_counts:
                                for family, count in user.token_counts.items():
                                    print(f"[RETRY] LOADING: User {token[:8]} {family}: {count.total} total, {count.requests} requests")
                            
                            self.users[token] = user
                            
                            # Check if user needs status field migration
                            if 'status' not in user_data:
                                users_to_migrate.append(token)
                                print(f"[RETRY] MIGRATION: User {token[:8]} needs status field migration")
                        except Exception as e:
                            print(f"Failed to load user {token}: {e}")
                
                print(f"[CAT] FIREBASE: Loaded {len(self.users)} users from Firebase")
                
                # Migrate users without status field
                if users_to_migrate:
                    print(f"[RETRY] MIGRATION: Migrating {len(users_to_migrate)} users to add status field")
                    for token in users_to_migrate:
                        self._migrate_user_status(token)
            
        except Exception as e:
            print(f"Failed to load users from Firebase: {e}")
    
    def _migrate_user_status(self, token: str):
        """Migrate user to add status field based on disabled_at"""
        try:
            if token not in self.users:
                return
            
            user = self.users[token]
            print(f"[RETRY] MIGRATION: Migrating user {token[:8]} - disabled_at: {user.disabled_at}")
            
            # Ensure status is set based on disabled_at
            if user.disabled_at:
                user.status = UserStatus.DISABLED
                print(f"[RETRY] MIGRATION: Set user {token[:8]} status to DISABLED")
            else:
                user.status = UserStatus.ACTIVE
                print(f"[RETRY] MIGRATION: Set user {token[:8]} status to ACTIVE")
            
            # Force immediate flush to update Firebase
            self._flush_to_firebase(token)
            print(f"[RETRY] MIGRATION: Completed migration for user {token[:8]}")
            
        except Exception as e:
            print(f"[ERROR] MIGRATION: Failed to migrate user {token[:8]}: {e}")
    
    def _sanitize_firebase_key(self, key: str) -> str:
        """Sanitize key for Firebase (remove invalid characters)"""
        return key.replace('.', '_').replace('#', '_').replace('$', '_').replace('[', '_').replace(']', '_')
    
    def _flush_to_firebase(self, token: str):
        """Flush user to Firebase"""
        if not self.firebase_db:
            print(f"[ERROR] FIREBASE: No Firebase DB connection for user {token[:8]}")
            return
        
        try:
            sanitized_token = self._sanitize_firebase_key(token)
            
            if token in self.users:
                # Update user
                user_data = self.users[token].to_dict()
                print(f"[CAT] FIREBASE: Flushing user {token[:8]} to Firebase")
                print(f"[CAT] FIREBASE: disabled_at in data: {user_data.get('disabled_at')}")
                print(f"[CAT] FIREBASE: disabled_reason in data: {user_data.get('disabled_reason')}")
                print(f"[CAT] FIREBASE: status in data: {user_data.get('status')}")
                print(f"[CAT] FIREBASE: user object status: {getattr(self.users[token], 'status', 'not set')}")
                print(f"[CAT] FIREBASE: user object is_disabled(): {self.users[token].is_disabled()}")
                print(f"[CAT] FIREBASE: token_counts in data: {list(user_data.get('token_counts', {}).keys())}")
                for family, count_data in user_data.get('token_counts', {}).items():
                    if isinstance(count_data, dict):
                        original_key = family.replace('_DOT_', '.').replace('_HASH_', '#').replace('_DOLLAR_', '$').replace('_LBRACKET_', '[').replace('_RBRACKET_', ']').replace('_SLASH_', '/')
                        print(f"[CAT] FIREBASE: {family} -> {original_key}: {count_data.get('total', 0)} total, {count_data.get('requests', 0)} requests")
                
                user_ref = self.firebase_db.child('users').child(sanitized_token)
                user_ref.set(user_data)
                print(f"[CAT] FIREBASE: Successfully flushed user {token[:8]} to Firebase")
            else:
                # Delete user
                print(f"[DELETE] FIREBASE: Deleting user {token[:8]} from Firebase")
                user_ref = self.firebase_db.child('users').child(sanitized_token)
                user_ref.delete()
        except Exception as e:
            print(f"[ERROR] FIREBASE: Failed to flush user {token[:8]} to Firebase: {e}")
            import traceback
            traceback.print_exc()
    
    def _batch_flush_to_firebase(self, tokens: List[str]):
        """Batch flush multiple users to Firebase for better performance"""
        if not self.firebase_db or not tokens:
            return
        
        try:
            print(f"[RETRY] BATCH FLUSH: Starting batch flush for {len(tokens)} users")
            # Prepare batch update data
            batch_data = {}
            to_delete = []
            
            for token in tokens:
                sanitized_token = self._sanitize_firebase_key(token)
                if token in self.users:
                    user = self.users[token]
                    user_data = user.to_dict()
                    
                    print(f"[RETRY] BATCH FLUSH: User {token[:8]} - disabled_at: {user.disabled_at}")
                    print(f"[RETRY] BATCH FLUSH: User {token[:8]} - disabled_reason: {user.disabled_reason}")
                    print(f"[RETRY] BATCH FLUSH: User {token[:8]} - status: {getattr(user, 'status', 'not set')}")
                    print(f"[RETRY] BATCH FLUSH: User {token[:8]} - to_dict status: {user_data.get('status')}")
                    print(f"[RETRY] BATCH FLUSH: User {token[:8]} - to_dict disabled_at: {user_data.get('disabled_at')}")
                    print(f"[RETRY] BATCH FLUSH: User {token[:8]} - is_disabled(): {user.is_disabled()}")
                    
                    batch_data[f'users/{sanitized_token}'] = user_data
                else:
                    # User was deleted
                    print(f"[DELETE] BATCH FLUSH: User {token[:8]} marked for deletion")
                    to_delete.append(sanitized_token)
            
            # Perform batch update for existing users
            if batch_data:
                print(f"[RETRY] BATCH FLUSH: Updating {len(batch_data)} users in Firebase")
                self.firebase_db.update(batch_data)
                print(f"[RETRY] BATCH FLUSH: Batch update completed")
            
            # Delete users that were removed
            for sanitized_token in to_delete:
                print(f"[DELETE] BATCH FLUSH: Deleting user from Firebase")
                user_ref = self.firebase_db.child('users').child(sanitized_token)
                user_ref.delete()
                
        except Exception as e:
            print(f"[ERROR] BATCH FLUSH: Failed to batch flush users to Firebase: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to individual flushes
            for token in tokens:
                self._flush_to_firebase(token)
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    print(f"[CLEAN] CLEANUP THREAD: Starting cleanup cycle")
                    
                    # Flush pending changes to Firebase in batches
                    with self.flush_queue_lock:
                        flush_tokens = list(self.flush_queue.copy())
                        self.flush_queue.clear()
                    
                    print(f"[CLEAN] CLEANUP THREAD: Found {len(flush_tokens)} users in flush queue")
                    
                    if flush_tokens:
                        # Process in batches for better performance
                        batch_size = 10
                        for i in range(0, len(flush_tokens), batch_size):
                            batch = flush_tokens[i:i + batch_size]
                            print(f"[CLEAN] CLEANUP THREAD: Processing batch {i//batch_size + 1} with {len(batch)} users")
                            self._batch_flush_to_firebase(batch)
                    
                    # Clean up expired temporary users
                    print(f"[CLEAN] CLEANUP THREAD: Running expired user cleanup")
                    self._cleanup_expired_users()
                    
                    print(f"[CLEAN] CLEANUP THREAD: Cleanup cycle completed, sleeping for 300 seconds")
                    time.sleep(300)  # Changed to 5 minutes for debugging
                except Exception as e:
                    print(f"[ERROR] CLEANUP THREAD: Cleanup thread error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(300)
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired_users(self):
        """Clean up expired temporary users"""
        with self.lock.write_lock():
            to_disable = []
            
            for token, user in self.users.items():
                if user.type == UserType.TEMPORARY and not user.is_disabled():
                    # Check if temporary user has expired (24 hours)
                    if user.created_at < datetime.now() - timedelta(hours=24):
                        to_disable.append(token)
            
            for token in to_disable:
                self._disable_user_internal(token, "Temporary token expired")
    
    def _hash_ip(self, ip: str) -> str:
        """Hash IP address for privacy"""
        return hashlib.sha256(ip.encode()).hexdigest()
    
    def _disable_user_internal(self, token: str, reason: str):
        """Internal method to disable user (assumes lock is held)"""
        if token in self.users:
            user = self.users[token]
            print(f"[CAT] DISABLE: User {token[:8]} before - disabled_at: {user.disabled_at}, status: {getattr(user, 'status', 'not set')}")
            
            user.disabled_at = datetime.now()
            user.disabled_reason = reason
            user.status = UserStatus.DISABLED  # Set explicit status
            
            print(f"[CAT] DISABLE: User {token[:8]} after setting - disabled_at: {user.disabled_at}, reason: {user.disabled_reason}, status: {user.status}")
            
            # Force immediate Firebase flush for critical operations
            try:
                print(f"[CAT] DISABLE: Immediately flushing user {token[:8]} to Firebase")
                self._flush_to_firebase(token)
                print(f"[CAT] DISABLE: Firebase flush completed for user {token[:8]}")
            except Exception as e:
                print(f"[ERROR] DISABLE: Firebase flush failed for user {token[:8]}: {e}")
            
            # Also add to flush queue for backup
            with self.flush_queue_lock:
                self.flush_queue.add(token)
    
    def create_user(self, user_type: UserType = UserType.NORMAL, 
                   prompt_limits: Optional[Dict[str, int]] = None,
                   nickname: Optional[str] = None,
                   temp_prompt_limits: Optional[int] = None,
                   max_ips: Optional[int] = None) -> str:
        """Create a new user and return their token"""
        with self.lock.write_lock():
            token = str(uuid.uuid4())
            
            user = User(
                token=token,
                type=user_type,
                nickname=nickname,
                prompt_limits=temp_prompt_limits,
                max_ips=max_ips
            )
            
            if prompt_limits:
                user.prompt_limits.update(prompt_limits)
            
            self.users[token] = user
            
            # Immediately save new user to Firebase to ensure instant availability
            if self.firebase_db:
                try:
                    sanitized_token = self._sanitize_firebase_key(token)
                    user_data = user.to_dict()
                    user_ref = self.firebase_db.child('users').child(sanitized_token)
                    user_ref.set(user_data)
                    print(f" IMMEDIATE SAVE: Successfully saved new user {token[:8]} to Firebase")
                except Exception as e:
                    print(f"[ERROR] IMMEDIATE SAVE: Failed to save new user {token[:8]} to Firebase: {e}")
                    # Add to flush queue as fallback
                    with self.flush_queue_lock:
                        self.flush_queue.add(token)
            else:
                # No Firebase connection, add to flush queue for later
                with self.flush_queue_lock:
                    self.flush_queue.add(token)
            
            return token
    
    def authenticate(self, token: str, ip: str) -> Tuple[AuthResult, Optional[User]]:
        """Authenticate user token and track IP"""
        with self.lock.write_lock():
            if token not in self.users:
                print(f"[ERROR] AUTH: User {token[:8]} not found in users dict")
                return AuthResult.NOT_FOUND, None
            
            user = self.users[token]
            print(f"[CAT] AUTH: Authenticating user {token[:8]}")
            print(f"[CAT] AUTH: User {token[:8]} disabled_at: {user.disabled_at}")
            print(f"[CAT] AUTH: User {token[:8]} disabled_reason: {user.disabled_reason}")
            print(f"[CAT] AUTH: User {token[:8]} status: {getattr(user, 'status', 'not set')}")
            print(f"[CAT] AUTH: User {token[:8]} hasattr status: {hasattr(user, 'status')}")
            print(f"[CAT] AUTH: User {token[:8]} status type: {type(getattr(user, 'status', None))}")
            print(f"[CAT] AUTH: User {token[:8]} is_disabled(): {user.is_disabled()}")
            print(f"[CAT] AUTH: User {token[:8]} to_dict status: {user.to_dict().get('status')}")
            print(f"[CAT] AUTH: User {token[:8]} to_dict disabled_at: {user.to_dict().get('disabled_at')}")
            
            # DISABLED: Firebase consistency check was causing corruption by reloading users
            # This was overwriting good in-memory data with potentially incomplete Firebase data
            # TODO: Implement a safer consistency check that doesn't corrupt user data
            print(f" FIREBASE CHECK: Consistency check temporarily disabled to prevent corruption")
            
            # Check if user is disabled (after potential reload)
            print(f"[SEARCH] AUTH: Final check - User {token[:8]} status: {getattr(user, 'status', 'not set')}")
            print(f"[SEARCH] AUTH: Final check - User {token[:8]} disabled_at: {user.disabled_at}")
            print(f"[SEARCH] AUTH: Final check - User {token[:8]} is_disabled(): {user.is_disabled()}")
            
            if user.is_disabled():
                print(f"[ERROR] AUTH: User {token[:8]} is disabled, returning DISABLED with reason: {user.disabled_reason}")
                return AuthResult.DISABLED, user
            else:
                print(f"[OK] AUTH: User {token[:8]} is NOT disabled, proceeding with authentication")
            
            # Check if temporary user has exceeded prompt limits
            if user.type == UserType.TEMPORARY and user.is_prompt_limit_exceeded():
                self._disable_user_internal(token, "Prompt limit exceeded")
                return AuthResult.DISABLED, user
            
            # Hash IP for privacy
            hashed_ip = self._hash_ip(ip)
            
            # Check if IP limit exceeded
            if hashed_ip not in user.ip and user.is_ip_limit_exceeded():
                if auth_config.max_ips_auto_ban:
                    self._disable_user_internal(token, "IP address limit exceeded")
                    return AuthResult.DISABLED, user
                else:
                    return AuthResult.LIMITED, user
            
            # Add IP if not already tracked
            if hashed_ip not in user.ip:
                user.ip.append(hashed_ip)
                user.ip_usage.append(IPUsage(ip=hashed_ip, last_used=datetime.now()))
            else:
                # Update existing IP usage
                for usage in user.ip_usage:
                    if usage.ip == hashed_ip:
                        usage.last_used = datetime.now()
                        break
            
            # Update last used
            user.last_used = datetime.now()
            with self.flush_queue_lock:
                self.flush_queue.add(token)
            
            return AuthResult.SUCCESS, user
    
    def increment_token_count(self, token: str, model_family: str, 
                            input_tokens: int, output_tokens: int, cost: float = 0.0) -> bool:
        """Increment token count for user and model family (legacy method)"""
        with self.lock.write_lock():
            if token not in self.users:
                return False
            
            user = self.users[token]
            
            if user.is_disabled():
                return False
            
            # Get or create token count for model family
            if model_family not in user.token_counts:
                user.token_counts[model_family] = TokenCount()
            
            # Use the new add_usage method
            user.token_counts[model_family].add_usage(input_tokens, output_tokens, cost)
            
            # Update general user stats (legacy method - main tracking done in add_request_tracking)
            user.total_cost += cost
            user.last_used = datetime.now()
            
            # Update IP usage if we can determine current IP
            # Note: This is a fallback method, prefer using add_request_tracking
            if user.ip_usage and len(user.ip_usage) > 0:
                # Update the most recently used IP
                latest_ip_usage = max(user.ip_usage, key=lambda x: x.last_used or datetime.min)
                latest_ip_usage.add_request(model_family, input_tokens + output_tokens)
            
            with self.flush_queue_lock:
                self.flush_queue.add(token)
            return True
    
    def disable_user(self, token: str, reason: str = None) -> bool:
        """Disable a user account"""
        with self.lock.write_lock():
            if token not in self.users:
                return False
            
            self._disable_user_internal(token, reason or "Disabled by admin")
            return True
    
    def reactivate_user(self, token: str) -> bool:
        """Reactivate a disabled user"""
        with self.lock.write_lock():
            if token not in self.users:
                print(f"[ERROR] REACTIVATE: User {token[:8]} not found in users dict")
                return False
            
            user = self.users[token]
            print(f"[CAT] REACTIVATE: User {token[:8]} before - disabled_at: {user.disabled_at}, reason: {user.disabled_reason}, status: {getattr(user, 'status', 'not set')}")
            
            # Clear disabled status
            user.disabled_at = None
            user.disabled_reason = None
            user.status = UserStatus.ACTIVE  # Set explicit status to active
            
            print(f"[CAT] REACTIVATE: User {token[:8]} after clearing - disabled_at: {user.disabled_at}, reason: {user.disabled_reason}, status: {user.status}")
            print(f"[CAT] REACTIVATE: User {token[:8]} is_disabled() after clearing: {user.is_disabled()}")
            
            # Verify the disabled_at field is actually None
            if user.disabled_at is not None:
                print(f"[ERROR] CRITICAL: User {token[:8]} disabled_at should be None but is: {user.disabled_at}")
            if user.disabled_reason is not None:
                print(f"[ERROR] CRITICAL: User {token[:8]} disabled_reason should be None but is: {user.disabled_reason}")
            if user.is_disabled():
                print(f"[ERROR] CRITICAL: User {token[:8]} is_disabled() should be False but returns True!")
            
            # Force immediate Firebase flush for critical operations
            try:
                print(f"[CAT] REACTIVATE: Immediately flushing user {token[:8]} to Firebase")
                self._flush_to_firebase(token)
                print(f"[CAT] REACTIVATE: Firebase flush completed for user {token[:8]}")
            except Exception as e:
                print(f"[ERROR] REACTIVATE: Firebase flush failed for user {token[:8]}: {e}")
            
            # Also add to flush queue for backup
            with self.flush_queue_lock:
                self.flush_queue.add(token)
            
            return True
    
    def delete_user(self, token: str) -> bool:
        """Permanently delete a user (must be disabled first)"""
        with self.lock.write_lock():
            if token not in self.users:
                return False
            
            user = self.users[token]
            if not user.is_disabled():
                raise ValueError("User must be disabled before deletion")
            
            del self.users[token]
            
            # Immediately delete from Firebase to prevent resurrection on server restart
            if self.firebase_db:
                try:
                    sanitized_token = self._sanitize_firebase_key(token)
                    user_ref = self.firebase_db.child('users').child(sanitized_token)
                    user_ref.delete()
                    print(f"[DELETE] IMMEDIATE DELETE: Successfully deleted user {token[:8]} from Firebase")
                except Exception as e:
                    print(f"[ERROR] IMMEDIATE DELETE: Failed to delete user {token[:8]} from Firebase: {e}")
                    # Add to flush queue as fallback
                    with self.flush_queue_lock:
                        self.flush_queue.add(token)
            else:
                # No Firebase connection, add to flush queue for later
                with self.flush_queue_lock:
                    self.flush_queue.add(token)
            
            return True
    
    def rotate_user_token(self, old_token: str) -> Optional[str]:
        """Rotate user token (generate new token, preserve data)"""
        with self.lock.write_lock():
            if old_token not in self.users:
                return None
            
            user = self.users[old_token]
            new_token = str(uuid.uuid4())
            
            # Update token in user object
            user.token = new_token
            
            # Move user to new token key
            self.users[new_token] = user
            del self.users[old_token]
            
            # Queue both for Firebase update
            self.flush_queue.add(old_token)  # Delete old
            self.flush_queue.add(new_token)  # Add new
            
            return new_token
    
    def get_user(self, token: str) -> Optional[User]:
        """Get user by token"""
        with self.lock.read_lock():
            return self.users.get(token)
    
    def get_all_users(self) -> List[User]:
        """Get all users"""
        with self.lock.read_lock():
            return list(self.users.values())
    
    def get_user_count(self) -> int:
        """Get total user count"""
        with self.lock.read_lock():
            return len(self.users)
    
    def check_quota(self, token: str, model_family: str) -> Tuple[bool, int, Optional[int]]:
        """Check if user has quota remaining. Returns (has_quota, used_requests, request_limit)"""
        with self.lock.read_lock():
            if token not in self.users:
                return False, 0, 0
            
            user = self.users[token]
            if user.is_disabled():
                return False, 0, 0
            
            used_requests = user.get_token_usage(model_family).requests
            request_limit = user.get_request_limit(model_family)
            
            # None means unlimited
            if request_limit is None:
                return True, used_requests, None
            
            return used_requests < request_limit, used_requests, request_limit

# Global user store instance
user_store = UserStore()