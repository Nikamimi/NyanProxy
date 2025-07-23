"""
FastAPI Security Module for NyanProxy

Handles authentication, authorization, and security middleware
Will eventually replace Flask-based authentication system
"""
import os
import hashlib
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader
import time

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_client_ip(request: Request) -> str:
    """Get client IP address from request headers"""
    # Check forwarded headers first
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"


async def authenticate_request(
    request: Request,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_scheme)
) -> Dict[str, Any]:
    """
    Authenticate incoming request using multiple methods
    
    This is a simplified version - full implementation will integrate with
    the existing user_store and authentication system from Flask app
    """
    client_ip = await get_client_ip(request)
    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
    
    # Check for API key authentication
    if api_key:
        # Validate against proxy keys (simplified for now)
        proxy_key = os.getenv('PROXY_KEY')
        if proxy_key and api_key == proxy_key:
            return {
                "type": "proxy_key",
                "authenticated": True,
                "ip": client_ip,
                "ip_hash": ip_hash,
                "user_agent": request.headers.get('User-Agent', '')
            }
    
    # Check for bearer token authentication
    if bearer_token:
        token = bearer_token.credentials
        # TODO: Integrate with existing user_store system
        # For now, this is a placeholder
        return {
            "type": "user_token",
            "token": token,
            "authenticated": True,
            "ip": client_ip,
            "ip_hash": ip_hash,
            "user_agent": request.headers.get('User-Agent', '')
        }
    
    # Check auth mode for open access
    auth_mode = os.getenv('AUTH_MODE', 'user_token')
    if auth_mode == 'open_access':
        return {
            "type": "open_access",
            "authenticated": True,
            "ip": client_ip,
            "ip_hash": ip_hash,
            "user_agent": request.headers.get('User-Agent', '')
        }
    
    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_auth(auth_data: Dict[str, Any] = Depends(authenticate_request)) -> Dict[str, Any]:
    """Dependency that requires authentication"""
    if not auth_data.get("authenticated"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )
    return auth_data


async def optional_auth(
    request: Request,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_scheme)
) -> Optional[Dict[str, Any]]:
    """Optional authentication - returns None if not authenticated"""
    try:
        return await authenticate_request(request, bearer_token, api_key)
    except HTTPException:
        return None


class RateLimiter:
    """Simple rate limiting class"""
    
    def __init__(self):
        self.requests = {}  # ip -> [(timestamp, count), ...]
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def is_rate_limited(self, ip: str, limit: int = 100, window: int = 60) -> bool:
        """Check if IP is rate limited"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_entries(current_time - window)
            self.last_cleanup = current_time
        
        # Count requests from this IP in the time window
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Remove old requests outside the window
        self.requests[ip] = [
            (timestamp, count) for timestamp, count in self.requests[ip]
            if current_time - timestamp < window
        ]
        
        # Count total requests in window
        total_requests = sum(count for _, count in self.requests[ip])
        
        if total_requests >= limit:
            return True
        
        # Add current request
        self.requests[ip].append((current_time, 1))
        return False
    
    def cleanup_old_entries(self, cutoff_time: float):
        """Remove old entries to prevent memory leaks"""
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                (timestamp, count) for timestamp, count in self.requests[ip]
                if timestamp > cutoff_time
            ]
            if not self.requests[ip]:
                del self.requests[ip]


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request) -> None:
    """Check rate limiting for request"""
    client_ip = await get_client_ip(request)
    
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )