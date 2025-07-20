from flask import request, jsonify, g, session, redirect, url_for
from functools import wraps
from typing import Optional, Tuple
import time
import hashlib
from collections import defaultdict
import secrets

from ..config.auth import auth_config
from ..services.user_store import user_store, AuthResult

# Rate limiting storage
rate_limit_store = defaultdict(list)

def extract_token_from_request() -> Optional[str]:
    """Extract authentication token from request headers"""
    # Check Authorization header (Bearer token)
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]  # Remove "Bearer " prefix
    
    # Check x-api-key header
    api_key = request.headers.get('x-api-key')
    if api_key:
        return api_key
    
    # Check key query parameter
    key_param = request.args.get('key')
    if key_param:
        return key_param
    
    return None

def get_client_ip() -> str:
    """Get client IP address from request"""
    # Check X-Forwarded-For header (for proxies)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    
    # Check X-Real-IP header (for nginx)
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fallback to remote_addr
    return request.remote_addr or '127.0.0.1'

def check_rate_limit(identifier: str, limit_per_minute: int = None) -> Tuple[bool, int]:
    """Check if identifier is within rate limit. Returns (allowed, remaining)"""
    if not auth_config.rate_limit_enabled:
        return True, limit_per_minute or auth_config.rate_limit_per_minute
    
    if limit_per_minute is None:
        limit_per_minute = auth_config.rate_limit_per_minute
    
    current_time = time.time()
    minute_ago = current_time - 60
    
    # Clean old entries
    rate_limit_store[identifier] = [
        timestamp for timestamp in rate_limit_store[identifier] 
        if timestamp > minute_ago
    ]
    
    # Check if under limit
    current_count = len(rate_limit_store[identifier])
    if current_count >= limit_per_minute:
        return False, 0
    
    # Add current request
    rate_limit_store[identifier].append(current_time)
    return True, limit_per_minute - current_count - 1

def authenticate_request() -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Authenticate request based on auth mode.
    Returns (success, error_message, user_data)
    """
    client_ip = get_client_ip()
    
    # Check authentication mode
    if auth_config.mode == "none":
        # No authentication required
        return True, None, {"type": "none", "ip": client_ip}
    
    elif auth_config.mode == "proxy_key":
        # Single proxy password authentication
        token = extract_token_from_request()
        
        if not token:
            return False, "Authentication required: provide API key", None
        
        if token != auth_config.proxy_password:
            return False, "Invalid API key", None
        
        # Check rate limit by IP for proxy key mode
        allowed, remaining = check_rate_limit(client_ip)
        if not allowed:
            return False, "Rate limit exceeded", None
        
        return True, None, {
            "type": "proxy_key",
            "ip": client_ip,
            "rate_limit_remaining": remaining
        }
    
    elif auth_config.mode == "user_token":
        # Individual user token authentication
        token = extract_token_from_request()
        
        if not token:
            return False, "Authentication required: provide user token", None
        
        # Authenticate with user store
        auth_result, user = user_store.authenticate(token, client_ip)
        
        if auth_result == AuthResult.NOT_FOUND:
            return False, "Invalid user token", None
        
        elif auth_result == AuthResult.DISABLED:
            reason = user.disabled_reason or "Account disabled"
            return False, f"Account disabled: {reason}", None
        
        elif auth_result == AuthResult.LIMITED:
            return False, "IP address limit exceeded", None
        
        elif auth_result == AuthResult.SUCCESS:
            # Check rate limit by user token
            allowed, remaining = check_rate_limit(token)
            if not allowed:
                user.record_rate_limit_violation()
                return False, "Rate limit exceeded", None
            
            return True, None, {
                "type": "user_token",
                "token": token,
                "user": user,
                "ip": client_ip,
                "rate_limit_remaining": remaining
            }
    
    return False, "Invalid authentication mode", None

def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        success, error_message, user_data = authenticate_request()
        
        if not success:
            return jsonify({"error": error_message}), 401
        
        # Store auth data in Flask g object for use in endpoint
        g.auth_data = user_data
        
        return f(*args, **kwargs)
    
    return decorated_function

def require_admin_auth(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for admin key in Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Admin authentication required"}), 401
        
        admin_key = auth_header[7:]  # Remove "Bearer " prefix
        
        if admin_key != auth_config.admin_key:
            return jsonify({"error": "Invalid admin key"}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function

def check_quota(model_family: str) -> Tuple[bool, Optional[str]]:
    """Check if current user has quota for model family"""
    if not hasattr(g, 'auth_data') or g.auth_data["type"] != "user_token":
        return True, None  # No quota limits for non-user-token auth
    
    token = g.auth_data["token"]
    has_quota, used, limit = user_store.check_quota(token, model_family)
    
    if not has_quota:
        limit_text = "unlimited" if limit is None else str(limit)
        return False, f"Request quota exceeded for {model_family}: {used}/{limit_text} requests used"
    
    return True, None

def track_token_usage(model_name: str, input_tokens: int, output_tokens: int, cost: float = 0.0, response_time_ms: float = 0.0):
    """Track token usage for current user with enhanced tracking"""
    if not hasattr(g, 'auth_data') or g.auth_data["type"] != "user_token":
        return  # No tracking for non-user-token auth
    
    token = g.auth_data["token"]
    user = g.auth_data["user"]
    ip_hash = hashlib.sha256(g.auth_data["ip"].encode()).hexdigest()
    user_agent = request.headers.get('User-Agent', '')
    
    # Determine model family from model name for backwards compatibility
    model_family = "openai"  # default
    if "claude" in model_name.lower():
        model_family = "anthropic"
    elif "gemini" in model_name.lower():
        model_family = "google"
    elif "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower():
        model_family = "openai"
    
    # Use the enhanced tracking method with specific model name
    user.add_request_tracking(
        model_family=model_name,  # Use specific model name for individual tracking
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        ip_hash=ip_hash,
        user_agent=user_agent
    )
    
    # Debug: Print token counts after tracking
    print(f"🔍 TRACKING: User {token[:8]} after tracking {model_name}:")
    if hasattr(user, 'token_counts') and user.token_counts:
        for family, count in user.token_counts.items():
            print(f"🔍 TRACKING: {family}: {count.total} total, {count.requests} requests, {count.input} input, {count.output} output")
    else:
        print(f"🔍 TRACKING: No token_counts found for user {token[:8]}")
    
    # Record successful prompt for happiness tracking
    user.record_successful_prompt()
    
    # Also use the new structured event logger
    from ..services.firebase_logger import structured_logger
    structured_logger.log_chat_completion(
        user_token=token,
        model_family=model_family,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        response_time_ms=response_time_ms,
        success=True,
        ip_hash=ip_hash,
        user_agent=user_agent
    )
    
    # Mark user for Firebase sync (immediate flush for token usage)
    user_store.flush_queue.add(token)
    
    # Also do immediate flush for token usage to ensure persistence
    try:
        user_store._flush_to_firebase(token)
    except Exception as e:
        print(f"⚠️ WARNING: Immediate flush failed for token usage, will retry in cleanup cycle: {e}")

def require_admin_session(f):
    """Decorator to require admin session authentication for web interface"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_authenticated'):
            return redirect(url_for('admin.login'))
        return f(*args, **kwargs)
    return decorated_function

def generate_csrf_token():
    """Generate a CSRF token"""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return session['csrf_token']

def validate_csrf_token(token):
    """Validate CSRF token"""
    return token and token == session.get('csrf_token')

def csrf_protect(f):
    """Decorator to protect against CSRF attacks"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'POST':
            token = request.form.get('_csrf') or request.headers.get('X-CSRFToken')
            if not validate_csrf_token(token):
                return jsonify({'error': 'CSRF token validation failed'}), 403
        return f(*args, **kwargs)
    return decorated_function