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
            'models_used': self.models_used,
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
            models_used=data.get('models_used', {}),
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
    ip: List[str] = None
    ip_usage: List[IPUsage] = None
    token_counts: Dict[str, TokenCount] = None
    token_limits: Dict[str, int] = None
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
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.ip is None:
            self.ip = []
        if self.ip_usage is None:
            self.ip_usage = []
        if self.token_counts is None:
            self.token_counts = {}
        if self.token_limits is None:
            # Default limits per model family
            self.token_limits = {
                'openai': 100000,
                'anthropic': 100000,
                'google': 100000,
                'mistral': 100000
            }
        if self.preferred_models is None:
            self.preferred_models = {}
            
        # Initialize cat mood based on user type
        if self.type == UserType.SPECIAL:
            self.mood = "playful"
        elif self.type == UserType.TEMPORARY:
            self.mood = "sleepy"
        else:
            self.mood = "happy"
    
    def is_disabled(self) -> bool:
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
        """Get remaining prompts for temporary users"""
        if self.prompt_limits is None:
            return None
        
        total_prompts = sum(ip_usage.prompt_count for ip_usage in self.ip_usage)
        return max(0, self.prompt_limits - total_prompts)
    
    def get_token_usage(self, model_family: str) -> TokenCount:
        return self.token_counts.get(model_family, TokenCount())
    
    def get_token_limit(self, model_family: str) -> int:
        return self.token_limits.get(model_family, 100000)
    
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
        
        # Update favorite treat based on most used model
        most_used_model = max(self.preferred_models.items(), key=lambda x: x[1], default=('', 0))[0]
        treat_map = {
            'openai': 'Fish treats',
            'anthropic': 'Catnip',
            'google': 'Tuna',
            'mistral': 'Chicken bits'
        }
        self.favorite_treat = treat_map.get(most_used_model, 'Generic cat treats')
    
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
            "happy": "😸",
            "sleepy": "😴",
            "grumpy": "😾",
            "playful": "😼"
        }
        return mood_emojis.get(self.mood, "😺")
    
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
    
    def to_dict(self) -> dict:
        return {
            'token': self.token,
            'type': self.type.value,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'disabled_at': self.disabled_at.isoformat() if self.disabled_at else None,
            'disabled_reason': self.disabled_reason,
            'ip': self.ip,
            'ip_usage': [usage.to_dict() for usage in self.ip_usage],
            'token_counts': {k: v.to_dict() for k, v in self.token_counts.items()},
            'token_limits': self.token_limits,
            'nickname': self.nickname,
            'prompt_limits': self.prompt_limits,
            'max_ips': self.max_ips,
            
            # Enhanced tracking fields
            'total_requests': self.total_requests,
            'total_cost': self.total_cost,
            'rate_limit_hits': self.rate_limit_hits,
            'last_rate_limit': self.last_rate_limit.isoformat() if self.last_rate_limit else None,
            'suspicious_activity_count': self.suspicious_activity_count,
            'preferred_models': self.preferred_models,
            'average_tokens_per_request': self.average_tokens_per_request,
            'peak_requests_per_hour': self.peak_requests_per_hour,
            'favorite_user_agent': self.favorite_user_agent,
            
            # Cat-themed fields
            'mood': self.mood,
            'mood_emoji': self.get_mood_emoji(),
            'favorite_treat': self.favorite_treat,
            'paw_prints': self.paw_prints
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        # Handle backward compatibility for created_at
        created_at = data.get('created_at')
        if created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            # Use current time for legacy users without created_at
            created_at = datetime.now()
        
        # Handle backward compatibility for token_counts
        token_counts = {}
        for k, v in data.get('token_counts', {}).items():
            if isinstance(v, dict) and 'add_usage' not in v:
                # New format with to_dict/from_dict
                token_counts[k] = TokenCount.from_dict(v)
            else:
                # Old format - convert to new format
                token_counts[k] = TokenCount(
                    input=v.get('input', 0),
                    output=v.get('output', 0),
                    total=v.get('total', 0),
                    requests=v.get('requests', 0),
                    cost_usd=v.get('cost_usd', 0.0)
                )
        
        return cls(
            token=data['token'],
            type=UserType(data.get('type', 'normal')),
            created_at=created_at,
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            disabled_at=datetime.fromisoformat(data['disabled_at']) if data.get('disabled_at') else None,
            disabled_reason=data.get('disabled_reason'),
            ip=data.get('ip', []),
            ip_usage=[IPUsage.from_dict(usage) for usage in data.get('ip_usage', [])],
            token_counts=token_counts,
            token_limits=data.get('token_limits', {}),
            nickname=data.get('nickname'),
            prompt_limits=data.get('prompt_limits'),
            max_ips=data.get('max_ips'),
            
            # Enhanced tracking fields with backward compatibility
            total_requests=data.get('total_requests', 0),
            total_cost=data.get('total_cost', 0.0),
            rate_limit_hits=data.get('rate_limit_hits', 0),
            last_rate_limit=datetime.fromisoformat(data['last_rate_limit']) if data.get('last_rate_limit') else None,
            suspicious_activity_count=data.get('suspicious_activity_count', 0),
            preferred_models=data.get('preferred_models', {}),
            average_tokens_per_request=data.get('average_tokens_per_request', 0.0),
            peak_requests_per_hour=data.get('peak_requests_per_hour', 0),
            favorite_user_agent=data.get('favorite_user_agent'),
            
            # Cat-themed fields
            mood=data.get('mood', 'happy'),
            favorite_treat=data.get('favorite_treat'),
            paw_prints=data.get('paw_prints', 0)
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
                    for token, user_data in users_data.items():
                        try:
                            user = User.from_dict(user_data)
                            self.users[token] = user
                        except Exception as e:
                            print(f"Failed to load user {token}: {e}")
                
                # Loaded users from Firebase
            
        except Exception as e:
            print(f"Failed to load users from Firebase: {e}")
    
    def _sanitize_firebase_key(self, key: str) -> str:
        """Sanitize key for Firebase (remove invalid characters)"""
        return key.replace('.', '_').replace('#', '_').replace('$', '_').replace('[', '_').replace(']', '_')
    
    def _flush_to_firebase(self, token: str):
        """Flush user to Firebase"""
        if not self.firebase_db:
            return
        
        try:
            sanitized_token = self._sanitize_firebase_key(token)
            
            if token in self.users:
                # Update user
                user_ref = self.firebase_db.child('users').child(sanitized_token)
                user_ref.set(self.users[token].to_dict())
            else:
                # Delete user
                user_ref = self.firebase_db.child('users').child(sanitized_token)
                user_ref.delete()
        except Exception as e:
            print(f"Failed to flush user {token} to Firebase: {e}")
    
    def _batch_flush_to_firebase(self, tokens: List[str]):
        """Batch flush multiple users to Firebase for better performance"""
        if not self.firebase_db or not tokens:
            return
        
        try:
            # Prepare batch update data
            batch_data = {}
            to_delete = []
            
            for token in tokens:
                sanitized_token = self._sanitize_firebase_key(token)
                if token in self.users:
                    user = self.users[token]
                    batch_data[f'users/{sanitized_token}'] = user.to_dict()
                else:
                    # User was deleted
                    to_delete.append(sanitized_token)
            
            # Perform batch update for existing users
            if batch_data:
                self.firebase_db.update(batch_data)
            
            # Delete users that were removed
            for sanitized_token in to_delete:
                user_ref = self.firebase_db.child('users').child(sanitized_token)
                user_ref.delete()
                
        except Exception as e:
            print(f"Failed to batch flush users to Firebase: {e}")
            # Fallback to individual flushes
            for token in tokens:
                self._flush_to_firebase(token)
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    # Flush pending changes to Firebase in batches
                    with self.flush_queue_lock:
                        flush_tokens = list(self.flush_queue.copy())
                        self.flush_queue.clear()
                    
                    if flush_tokens:
                        # Process in batches for better performance
                        batch_size = 10
                        for i in range(0, len(flush_tokens), batch_size):
                            batch = flush_tokens[i:i + batch_size]
                            self._batch_flush_to_firebase(batch)
                    
                    # Clean up expired temporary users
                    self._cleanup_expired_users()
                    
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    print(f"Cleanup thread error: {e}")
                    time.sleep(60)
        
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
            user.disabled_at = datetime.now()
            user.disabled_reason = reason
            with self.flush_queue_lock:
                self.flush_queue.add(token)
    
    def create_user(self, user_type: UserType = UserType.NORMAL, 
                   token_limits: Optional[Dict[str, int]] = None,
                   nickname: Optional[str] = None,
                   prompt_limits: Optional[int] = None,
                   max_ips: Optional[int] = None) -> str:
        """Create a new user and return their token"""
        with self.lock.write_lock():
            token = str(uuid.uuid4())
            
            user = User(
                token=token,
                type=user_type,
                nickname=nickname,
                prompt_limits=prompt_limits,
                max_ips=max_ips
            )
            
            if token_limits:
                user.token_limits.update(token_limits)
            
            self.users[token] = user
            with self.flush_queue_lock:
                self.flush_queue.add(token)
            
            return token
    
    def authenticate(self, token: str, ip: str) -> Tuple[AuthResult, Optional[User]]:
        """Authenticate user token and track IP"""
        with self.lock.write_lock():
            if token not in self.users:
                return AuthResult.NOT_FOUND, None
            
            user = self.users[token]
            
            # Check if user is disabled
            if user.is_disabled():
                return AuthResult.DISABLED, user
            
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
            
            # Update general user stats
            user.total_requests += 1
            user.total_cost += cost
            user.last_used = datetime.now()
            user.paw_prints += 1
            
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
                return False
            
            user = self.users[token]
            user.disabled_at = None
            user.disabled_reason = None
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
            with self.flush_queue_lock:
                self.flush_queue.add(token)  # This will trigger Firebase deletion
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
    
    def check_quota(self, token: str, model_family: str) -> Tuple[bool, int, int]:
        """Check if user has quota remaining. Returns (has_quota, used, limit)"""
        with self.lock.read_lock():
            if token not in self.users:
                return False, 0, 0
            
            user = self.users[token]
            if user.is_disabled():
                return False, 0, 0
            
            used = user.get_token_usage(model_family).total
            limit = user.get_token_limit(model_family)
            
            return used < limit, used, limit

# Global user store instance
user_store = UserStore()