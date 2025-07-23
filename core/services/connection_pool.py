"""
HTTP Connection Pool Management Service for NyanProxy

Manages HTTP connection pools for better performance under load
Extracted from core/app.py for better modularity
"""
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ConnectionPoolManager:
    """Manages HTTP connection pools for better performance under load"""
    
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        
    def get_session(self, service: str):
        """Get a session with connection pooling for a specific service"""
        with self.lock:
            if service not in self.sessions:
                session = requests.Session()
                
                # Configure retry strategy
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
                )
                
                # Configure connection pooling
                adapter = HTTPAdapter(
                    pool_connections=20,  # Number of connection pools to cache
                    pool_maxsize=20,      # Maximum number of connections per pool
                    max_retries=retry_strategy,
                    pool_block=False
                )
                
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # Note: timeout is set per request, not on session
                # Store default timeout values for this service
                session._default_timeout = (10, 30)  # (connect, read) timeout
                
                self.sessions[service] = session
                print(f"Created connection pool for {service}")
                
            return self.sessions[service]
    
    def close_all_sessions(self):
        """Close all session connection pools"""
        with self.lock:
            for service, session in self.sessions.items():
                try:
                    session.close()
                    print(f"Closed connection pool for {service}")
                except Exception as e:
                    print(f"Error closing session for {service}: {e}")
            self.sessions.clear()