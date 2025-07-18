import os
import base64
from typing import Optional, Literal
from dotenv import load_dotenv

AuthMode = Literal["none", "proxy_key", "user_token"]

class AuthConfig:
    """Authentication configuration with environment variable support"""
    
    def __init__(self):
        # Load .env from the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(project_root, '.env')
        load_dotenv(env_path)
        
        self.mode: AuthMode = os.getenv('AUTH_MODE', 'user_token').lower()
        self.proxy_password: Optional[str] = os.getenv('PROXY_PASSWORD')
        self.admin_key: Optional[str] = os.getenv('ADMIN_KEY')
        
        # User token settings
        self.max_ips_per_user: int = int(os.getenv('MAX_IPS_PER_USER', '3'))
        self.max_ips_auto_ban: bool = os.getenv('MAX_IPS_AUTO_BAN', 'true').lower() == 'true'
        
        # Rate limiting
        self.rate_limit_per_minute: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
        self.rate_limit_enabled: bool = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        
        # Firebase configuration
        self.firebase_url: Optional[str] = os.getenv('FIREBASE_URL')
        self.firebase_service_account_key: Optional[str] = self._get_firebase_service_account_key()
        
        # Validate configuration
        self._validate_config()
    
    def _get_firebase_service_account_key(self) -> Optional[str]:
        """Get Firebase service account key, supporting both JSON and Base64 formats"""
        key = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
        if not key:
            return None
        
        # Check if it's base64 encoded (try to decode)
        try:
            # Try to decode as base64 first
            decoded = base64.b64decode(key).decode('utf-8')
            # If successful and looks like JSON, return it
            if decoded.strip().startswith('{'):
                # Base64 key decoded successfully
                return decoded
        except Exception as e:
            # Base64 decode error - using key as-is
            pass
        
        # Return as-is (assume it's already JSON)
        return key
    
    def _validate_config(self):
        """Validate authentication configuration"""
        if self.mode not in ["none", "proxy_key", "user_token"]:
            raise ValueError(f"Invalid AUTH_MODE: {self.mode}. Must be 'none', 'proxy_key', or 'user_token'")
        
        if self.mode == "proxy_key" and not self.proxy_password:
            raise ValueError("PROXY_PASSWORD is required when AUTH_MODE is 'proxy_key'")
        
        if self.mode == "user_token" and not self.firebase_url:
            print("Warning: Firebase URL not configured. User tokens will be stored in memory only.")
    
    def is_auth_required(self) -> bool:
        """Check if authentication is required"""
        return self.mode in ["proxy_key", "user_token"]
    
    def is_user_token_mode(self) -> bool:
        """Check if user token mode is enabled"""
        return self.mode == "user_token"
    
    def is_proxy_key_mode(self) -> bool:
        """Check if proxy key mode is enabled"""
        return self.mode == "proxy_key"

# Global config instance
auth_config = AuthConfig()