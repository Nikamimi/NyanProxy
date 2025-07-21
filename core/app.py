from flask import Flask, request, jsonify, Response, render_template, g, session
import requests
import os
import json
import random
import time
import threading
import hashlib
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .health_checker import health_manager, HealthResult

# Import tokenizer libraries
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, token counting will be approximate")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    import os
    # Load .env from the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    # Loading .env file
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

from ai.tokenizers.unified_tokenizer import unified_tokenizer

# Import authentication system
from src.config.auth import auth_config
from src.middleware.auth import require_auth, check_quota, track_token_usage, get_client_ip, authenticate_request
from src.services.firebase_logger import event_logger, structured_logger
from src.services.user_store import user_store, AuthResult
from src.routes.admin import admin_bp
from src.routes.admin_web import admin_web_bp
from src.routes.model_families_admin import model_families_bp
from src.middleware.auth import generate_csrf_token
from src.services.model_families import model_manager, AIProvider

class ThreadManager:
    """Manages background threads and prevents resource leaks"""
    def __init__(self):
        self.active_threads = {}  # thread_id -> thread info
        self.executor_pools = {}  # pool_name -> ThreadPoolExecutor
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
    def create_thread_pool(self, pool_name: str, max_workers: int = 10):
        """Create a reusable thread pool"""
        with self.lock:
            if pool_name in self.executor_pools:
                return self.executor_pools[pool_name]
            
            pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{pool_name}_")
            self.executor_pools[pool_name] = pool
            print(f"🧵 Created thread pool '{pool_name}' with {max_workers} workers")
            return pool
    
    def submit_task(self, pool_name: str, func, *args, **kwargs):
        """Submit a task to a managed thread pool"""
        pool = self.create_thread_pool(pool_name)
        future = pool.submit(func, *args, **kwargs)
        return future
    
    def create_daemon_thread(self, target, name: str = None, args: tuple = ()):
        """Create a managed daemon thread with cleanup tracking"""
        thread_id = f"{name}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        def wrapped_target(*args):
            try:
                target(*args)
            except Exception as e:
                print(f"⚠️ Thread {thread_id} error: {e}")
            finally:
                with self.lock:
                    if thread_id in self.active_threads:
                        del self.active_threads[thread_id]
                        print(f"🧹 Cleaned up thread {thread_id}")
        
        thread = threading.Thread(
            target=wrapped_target,
            name=name or thread_id,
            args=args,
            daemon=True
        )
        
        with self.lock:
            self.active_threads[thread_id] = {
                'thread': thread,
                'created_at': time.time(),
                'name': name or thread_id
            }
        
        thread.start()
        print(f"🚀 Started managed thread: {thread_id}")
        return thread_id
    
    def cleanup_dead_threads(self):
        """Remove references to dead threads"""
        with self.lock:
            dead_threads = []
            for thread_id, info in self.active_threads.items():
                if not info['thread'].is_alive():
                    dead_threads.append(thread_id)
            
            for thread_id in dead_threads:
                del self.active_threads[thread_id]
                
            if dead_threads:
                print(f"🧹 Cleaned up {len(dead_threads)} dead threads")
    
    def get_thread_stats(self):
        """Get current thread statistics"""
        with self.lock:
            active_count = sum(1 for info in self.active_threads.values() 
                             if info['thread'].is_alive())
            
            return {
                'active_managed_threads': active_count,
                'total_managed_threads': len(self.active_threads),
                'thread_pools': list(self.executor_pools.keys()),
                'oldest_thread_age': time.time() - min(
                    (info['created_at'] for info in self.active_threads.values()),
                    default=time.time()
                )
            }
    
    def shutdown_all(self):
        """Gracefully shutdown all managed threads and pools"""
        print("🛑 Shutting down thread manager...")
        self.shutdown_event.set()
        
        # Shutdown thread pools
        for pool_name, pool in self.executor_pools.items():
            print(f"🛑 Shutting down pool: {pool_name}")
            pool.shutdown(wait=True, cancel_futures=True)
        
        print("✅ Thread manager shutdown complete")

# Global thread manager instance
thread_manager = ThreadManager()

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
                print(f"🔗 Created connection pool for {service}")
                
            return self.sessions[service]
    
    def close_all_sessions(self):
        """Close all session connection pools"""
        with self.lock:
            for service, session in self.sessions.items():
                try:
                    session.close()
                    print(f"🔗 Closed connection pool for {service}")
                except Exception as e:
                    print(f"⚠️ Error closing session for {service}: {e}")
            self.sessions.clear()

# Global connection pool manager
connection_pool = ConnectionPoolManager()

class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()  # Current process for monitoring
        self.request_counts = {
            'chat_completions': 0,
            'models': 0,
            'health': 0,
            'key_status': 0
        }
        self.error_counts = {
            'chat_completions': 0,
            'models': 0,
            'key_errors': 0
        }
        self.total_requests = 0
        self.last_request_time = None
        self.response_times = []
        # Token tracking
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Per-service tracking (dynamically populated as services are used)
        self.service_tokens = {}
        # IP tracking for active prompters
        self.active_ips = []  # List of (ip, timestamp) tuples
        # System monitoring
        self.peak_memory_mb = 0
        self.restart_count = 0
        self.thread_count_history = []
        self.gc_stats = {'collections': 0, 'collected': 0, 'uncollectable': 0}
        self.lock = threading.Lock()
    
    def track_request(self, endpoint: str, response_time: float = None, error: bool = False, tokens: dict = None, service: str = None):
        with self.lock:
            self.total_requests += 1
            self.last_request_time = datetime.now()
            
            if endpoint in self.request_counts:
                self.request_counts[endpoint] += 1
            
            if error and endpoint in self.error_counts:
                self.error_counts[endpoint] += 1
            
            if response_time is not None:
                self.response_times.append(response_time)
                # Keep only last 100 response times
                if len(self.response_times) > 100:
                    self.response_times.pop(0)
            
            # Always track requests, even if tokens is None
            if service:
                # Initialize service if not exists
                if service not in self.service_tokens:
                    self.service_tokens[service] = {
                        'successful_requests': 0,
                        'total_tokens': 0,
                        'response_times': []
                    }
                    # Initialized tracking for new service
                
                # Track successful requests (only count if not an error)
                if not error:
                    self.service_tokens[service]['successful_requests'] += 1
                    # Incremented successful requests
                
                # Track response times per service
                if response_time is not None:
                    self.service_tokens[service]['response_times'].append(response_time)
                    # Keep only last 100 response times per service
                    if len(self.service_tokens[service]['response_times']) > 100:
                        self.service_tokens[service]['response_times'].pop(0)
            
            # Track tokens if available
            if tokens and service:
                prompt_tokens = tokens.get('prompt_tokens', 0)
                completion_tokens = tokens.get('completion_tokens', 0)
                total_tokens = tokens.get('total_tokens', 0)
                
                # Tracking tokens
                
                # Update global counters
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                # Track total tokens (input + output)
                self.service_tokens[service]['total_tokens'] += total_tokens
            
            # Periodic memory check (every 50 requests)
            if self.total_requests % 50 == 0:
                self.check_memory_threshold()
    
    def track_ip(self, ip: str):
        """Track IP address for current prompter counting"""
        with self.lock:
            current_time = time.time()
            # Add current IP with timestamp
            self.active_ips.append((ip, current_time))
            # Clean up old entries (older than 60 seconds)
            cutoff_time = current_time - 60
            self.active_ips = [(ip, timestamp) for ip, timestamp in self.active_ips if timestamp > cutoff_time]
    
    def get_current_prompters(self):
        """Get count of unique IPs in the past 60 seconds"""
        # Note: This method should be called from within a locked context
        current_time = time.time()
        cutoff_time = current_time - 60
        # Clean up old entries
        self.active_ips = [(ip, timestamp) for ip, timestamp in self.active_ips if timestamp > cutoff_time]
        # Return unique IP count
        unique_ips = set(ip for ip, timestamp in self.active_ips)
        return len(unique_ips)
    
    def get_uptime(self):
        return time.time() - self.start_time
    
    def get_average_response_time(self):
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)
    
    def get_system_stats(self):
        """Get current system statistics"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Update peak memory
            if memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = memory_mb
            
            # Get thread count
            thread_count = self.process.num_threads()
            self.thread_count_history.append(thread_count)
            
            # Keep only last 100 thread counts
            if len(self.thread_count_history) > 100:
                self.thread_count_history.pop(0)
            
            # Get CPU usage (non-blocking)
            cpu_percent = self.process.cpu_percent(interval=None)
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            if gc_stats:
                self.gc_stats['collections'] = sum(stat.get('collections', 0) for stat in gc_stats)
                self.gc_stats['collected'] = sum(stat.get('collected', 0) for stat in gc_stats)
                self.gc_stats['uncollectable'] = sum(stat.get('uncollectable', 0) for stat in gc_stats)
            
            return {
                'memory_mb': round(memory_mb, 2),
                'peak_memory_mb': round(self.peak_memory_mb, 2),
                'cpu_percent': round(cpu_percent, 1),
                'thread_count': thread_count,
                'avg_thread_count': round(sum(self.thread_count_history) / len(self.thread_count_history), 1) if self.thread_count_history else 0,
                'gc_stats': self.gc_stats.copy(),
                'open_file_descriptors': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0
            }
        except Exception as e:
            print(f"⚠️ Error getting system stats: {e}")
            return {
                'memory_mb': 0,
                'peak_memory_mb': 0,
                'cpu_percent': 0,
                'thread_count': 0,
                'avg_thread_count': 0,
                'gc_stats': {'collections': 0, 'collected': 0, 'uncollectable': 0},
                'open_file_descriptors': 0
            }
    
    def force_garbage_collection(self):
        """Force garbage collection and return collected objects count"""
        try:
            collected = gc.collect()
            print(f"🧹 Garbage collection: {collected} objects collected")
            return collected
        except Exception as e:
            print(f"⚠️ Error during garbage collection: {e}")
            return 0
    
    def check_memory_threshold(self, threshold_mb=500):
        """Check if memory usage exceeds threshold and force GC if needed"""
        try:
            current_memory = self.process.memory_info().rss / (1024 * 1024)
            if current_memory > threshold_mb:
                print(f"🚨 Memory usage high: {current_memory:.2f}MB, forcing garbage collection")
                collected = self.force_garbage_collection()
                return True, collected
            return False, 0
        except Exception as e:
            print(f"⚠️ Error checking memory threshold: {e}")
            return False, 0
    
    def get_metrics(self):
        with self.lock:
            system_stats = self.get_system_stats()
            thread_stats = thread_manager.get_thread_stats()
            
            # Periodic cleanup of dead threads
            thread_manager.cleanup_dead_threads()
            
            return {
                'uptime_seconds': self.get_uptime(),
                'total_requests': self.total_requests,
                'request_counts': self.request_counts.copy(),
                'error_counts': self.error_counts.copy(),
                'last_request': self.last_request_time.isoformat() if self.last_request_time else None,
                'average_response_time': self.get_average_response_time(),
                'total_tokens': self.total_tokens,
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'service_tokens': self.service_tokens.copy(),
                'current_prompters': self.get_current_prompters(),
                'system_stats': system_stats,
                'thread_stats': thread_stats,
                'restart_count': self.restart_count
            }



class APIKeyManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.key_usage = {}  # Track usage per key for rate limiting
        self.failed_keys = set()  # Track temporarily failed keys
        self.key_health = {}  # Track detailed health status for each key
        self.lock = threading.Lock()  # Thread safety for key health updates
        self._initialize_key_health()
    
    def _parse_keys(self, key_string: str) -> List[str]:
        """Parse comma and line-separated keys"""
        if not key_string:
            return []
        
        keys = []
        # Split by both commas and newlines
        for line in key_string.replace(',', '\n').split('\n'):
            key = line.strip()
            if key and key not in keys:
                keys.append(key)
        return keys
    
    def _load_api_keys(self) -> Dict[str, List[str]]:
        """Load API keys from environment variables or config"""
        keys = {}
        
        # OpenAI keys - support comma/line separated format
        openai_keys = []
        
        # Try comma/line separated format first
        bulk_key = os.getenv('OPENAI_API_KEYS')
        if bulk_key:
            parsed_keys = self._parse_keys(bulk_key)
            # Found OpenAI keys
            openai_keys.extend(parsed_keys)
        
        # Fallback to single key
        if not openai_keys:
            single_key = os.getenv('OPENAI_API_KEY')
            if single_key:
                openai_keys.extend(self._parse_keys(single_key))
        
        # Only add services that have actual keys
        if openai_keys:
            keys['openai'] = list(set(openai_keys))  # Remove duplicates
        
        # Anthropic keys
        anthropic_keys = []
        bulk_key = os.getenv('ANTHROPIC_API_KEYS')
        if bulk_key:
            anthropic_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('ANTHROPIC_API_KEY')
        if single_key:
            anthropic_keys.extend(self._parse_keys(single_key))
        
        # Only add anthropic if it has keys
        if anthropic_keys:
            keys['anthropic'] = list(set(anthropic_keys))
        
        # Google keys
        google_keys = []
        bulk_key = os.getenv('GOOGLE_API_KEYS')
        if bulk_key:
            google_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('GOOGLE_API_KEY')
        if single_key:
            google_keys.extend(self._parse_keys(single_key))
        
        if google_keys:
            keys['google'] = list(set(google_keys))
        
        # Mistral keys
        mistral_keys = []
        bulk_key = os.getenv('MISTRAL_API_KEYS')
        if bulk_key:
            mistral_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('MISTRAL_API_KEY')
        if single_key:
            mistral_keys.extend(self._parse_keys(single_key))
        
        if mistral_keys:
            keys['mistral'] = list(set(mistral_keys))
        
        # Groq keys
        groq_keys = []
        bulk_key = os.getenv('GROQ_API_KEYS')
        if bulk_key:
            groq_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('GROQ_API_KEY')
        if single_key:
            groq_keys.extend(self._parse_keys(single_key))
        
        if groq_keys:
            keys['groq'] = list(set(groq_keys))
        
        # Cohere keys
        cohere_keys = []
        bulk_key = os.getenv('COHERE_API_KEYS')
        if bulk_key:
            cohere_keys.extend(self._parse_keys(bulk_key))
        
        single_key = os.getenv('COHERE_API_KEY')
        if single_key:
            cohere_keys.extend(self._parse_keys(single_key))
        
        if cohere_keys:
            keys['cohere'] = list(set(cohere_keys))
        
        return keys
    
    def _initialize_key_health(self):
        """Initialize health tracking for all keys"""
        with self.lock:
            for service, keys in self.api_keys.items():
                for key in keys:
                    if key and key not in self.key_health:
                        self.key_health[key] = {
                            'status': 'unknown',
                            'service': service,
                            'last_error': None,
                            'last_success': None,
                            'last_failure': None,
                            'failure_count': 0,
                            'consecutive_failures': 0,
                            'error_type': None
                        }
    
    def perform_proactive_health_check(self, service: str, api_key: str):
        """Perform proactive health check using the health checker system"""
        try:
            result = health_manager.check_service_health(service, api_key)
            
            # Update key health based on result
            if result.status == 'healthy':
                self.update_key_health(api_key, True)
            else:
                self.update_key_health(
                    api_key, 
                    False, 
                    result.status_code, 
                    result.error_message
                )
                
        except Exception as e:
            # If health check fails, mark as network error
            self.update_key_health(api_key, False, None, f"Health check failed: {str(e)}")
    
    def run_initial_health_checks(self):
        """Run initial health checks for all keys concurrently in background"""
        def check_all_keys():
            # Collect all key-service pairs for concurrent execution
            key_service_pairs = []
            for service, keys in self.api_keys.items():
                for key in keys:
                    if key:
                        key_service_pairs.append((service, key))
            
            if not key_service_pairs:
                return
            
            print(f"🚀 Starting concurrent health checks for {len(key_service_pairs)} API keys...")
            start_time = time.time()
            
            # Use managed thread pool for health checks
            health_pool = thread_manager.create_thread_pool("health_checks", max_workers=min(30, len(key_service_pairs)))
            
            # Submit all health check tasks
            futures = []
            for service, key in key_service_pairs:
                future = health_pool.submit(self.perform_proactive_health_check, service, key)
                futures.append((future, service, key))
            
            # Process completed tasks
            completed = 0
            for future, service, key in futures:
                try:
                    future.result(timeout=30)  # Add timeout
                    completed += 1
                    
                    # Log progress every 20 completions or when significant milestones reached
                    if completed % 20 == 0 or completed in [10, 25, 50, 100]:
                        elapsed_partial = time.time() - start_time
                        rate = completed / elapsed_partial if elapsed_partial > 0 else 0
                        print(f"✅ Health checks: {completed}/{len(key_service_pairs)} ({rate:.1f}/sec)")
                        
                except Exception as e:
                    print(f"⚠️ Health check failed for {service} key {key[:8]}...: {e}")
                    completed += 1
            
            elapsed = time.time() - start_time
            print(f"🎉 All health checks completed in {elapsed:.2f}s ({len(key_service_pairs)/elapsed:.1f} keys/sec)")
        
        # Use managed daemon thread
        thread_manager.create_daemon_thread(check_all_keys, "health_check_runner")
    
    def _classify_error(self, status_code: int, error_message: str) -> tuple:
        """Classify error type and return (error_type, is_retryable)"""
        error_msg_lower = error_message.lower()
        
        # Authentication errors (not retryable with same key)
        if status_code in [401, 403]:
            return 'invalid_key', False
        
        # Rate limiting (retryable)
        if status_code == 429:
            return 'rate_limited', True
        
        # Quota/billing issues
        quota_indicators = ['quota exceeded', 'billing', 'insufficient_quota', 'usage limit']
        if any(indicator in error_msg_lower for indicator in quota_indicators):
            return 'quota_exceeded', False
        
        # Rate limit indicators in message
        rate_limit_indicators = [
            'rate limit', 'too many requests', 'rate_limit_exceeded',
            'requests per minute', 'rpm', 'tpm'
        ]
        if any(indicator in error_msg_lower for indicator in rate_limit_indicators):
            return 'rate_limited', True
        
        # Server errors (potentially retryable)
        if status_code >= 500:
            return 'server_error', True
        
        # Other client errors
        if status_code >= 400:
            return 'client_error', False
        
        return 'unknown_error', False
    
    def get_api_key(self, service: str, exclude_failed: bool = True) -> str:
        """Get next available API key for the service with rate limit handling"""
        if service not in self.api_keys or not self.api_keys[service]:
            return None
        
        available_keys = [key for key in self.api_keys[service] if key]
        
        # Remove failed keys if requested
        if exclude_failed:
            available_keys = [key for key in available_keys if key not in self.failed_keys]
        
        if not available_keys:
            # If all keys failed, try again with failed keys (maybe they recovered)
            if exclude_failed:
                return self.get_api_key(service, exclude_failed=False)
            return None
        
        # Simple round-robin selection
        if service not in self.key_usage:
            self.key_usage[service] = 0
        
        key_index = self.key_usage[service] % len(available_keys)
        self.key_usage[service] += 1
        
        return available_keys[key_index]
    
    def update_key_health(self, key: str, success: bool, status_code: int = None, error_message: str = None):
        """Update key health status based on API response"""
        with self.lock:
            if key not in self.key_health:
                # Initialize if not exists
                service = next((svc for svc, keys in self.api_keys.items() if key in keys), 'unknown')
                self.key_health[key] = {
                    'status': 'unknown',
                    'service': service,
                    'last_error': None,
                    'last_success': None,
                    'last_failure': None,
                    'failure_count': 0,
                    'consecutive_failures': 0,
                    'error_type': None
                }
            
            health = self.key_health[key]
            now = datetime.now()
            
            if success:
                health['status'] = 'healthy'
                health['last_success'] = now
                health['consecutive_failures'] = 0
                health['error_type'] = None
                health['last_error'] = None
            else:
                health['last_failure'] = now
                health['failure_count'] += 1
                health['consecutive_failures'] += 1
                
                if status_code and error_message:
                    error_type, is_retryable = self._classify_error(status_code, error_message)
                    health['error_type'] = error_type
                    health['status'] = error_type
                    health['last_error'] = error_message
                    
                    # Only mark as failed for retryable errors
                    if is_retryable:
                        self.failed_keys.add(key)
                        # Auto-recovery after some time
                        threading.Timer(300, lambda: self.failed_keys.discard(key)).start()
                else:
                    health['status'] = 'failed'
                    health['error_type'] = 'unknown_error'
    
    def mark_key_failed(self, service: str, key: str):
        """Mark a key as temporarily failed (legacy method)"""
        self.failed_keys.add(key)
        # Auto-recovery after some time (simplified)
        threading.Timer(300, lambda: self.failed_keys.discard(key)).start()
    
    def handle_api_error(self, service: str, key: str, error_message: str, status_code: int = None) -> bool:
        """Handle API error and return True if should retry with different key"""
        # Update key health with failure
        self.update_key_health(key, False, status_code, error_message)
        
        # Check if error is retryable
        if status_code and error_message:
            error_type, is_retryable = self._classify_error(status_code, error_message)
            return is_retryable
        
        # Legacy fallback for rate limiting
        rate_limit_indicators = [
            'rate limit', 'too many requests', 'quota exceeded',
            'rate_limit_exceeded', 'requests per minute', 'rpm', 'tpm'
        ]
        if any(indicator in error_message.lower() for indicator in rate_limit_indicators):
            self.mark_key_failed(service, key)
            return True
        
        return False

key_manager = APIKeyManager()
metrics = MetricsTracker()
app = Flask(__name__, template_folder='../pages', static_folder='../static')

# Startup logging and restart detection
startup_time = datetime.now()
print(f"🐱 NyanProxy starting up at {startup_time.isoformat()}")
print(f"🔧 Process ID: {os.getpid()}")
print(f"🧵 Main thread ID: {threading.current_thread().ident}")

# Check for previous crash/restart indicators
restart_file = os.path.join(os.path.dirname(__file__), '..', '.restart_count')
try:
    if os.path.exists(restart_file):
        with open(restart_file, 'r') as f:
            restart_count = int(f.read().strip()) + 1
            metrics.restart_count = restart_count
        print(f"🔄 Detected restart #{restart_count}")
    else:
        restart_count = 1
        metrics.restart_count = restart_count
        print("✨ First startup detected")
    
    # Update restart count file
    with open(restart_file, 'w') as f:
        f.write(str(restart_count))
    
    # Log startup event to structured logger if available
    try:
        structured_logger.log_system_event(
            event_type='startup',
            details={
                'restart_count': restart_count,
                'startup_time': startup_time.isoformat(),
                'process_id': os.getpid(),
                'thread_id': threading.current_thread().ident
            }
        )
    except Exception as e:
        print(f"⚠️ Could not log startup event: {e}")
        
except Exception as e:
    print(f"⚠️ Error handling restart detection: {e}")
    metrics.restart_count = 1

# Configure Flask
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key-change-in-production')
app.permanent_session_lifetime = timedelta(hours=24)  # Sessions last 24 hours

# Register admin blueprints
app.register_blueprint(admin_bp)
app.register_blueprint(admin_web_bp)
app.register_blueprint(model_families_bp)

# Add CSRF token to template context
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf_token)

# Load anti-abuse configuration from Firebase at startup
try:
    from src.services.config_manager import config_manager
    anti_abuse_config = config_manager.load_anti_abuse_config()
    
    # Update auth_config with persisted Firebase values
    auth_config.rate_limit_enabled = anti_abuse_config.rate_limit_enabled
    auth_config.rate_limit_per_minute = anti_abuse_config.rate_limit_per_minute
    auth_config.max_ips_per_user = anti_abuse_config.max_ips_per_user
    auth_config.max_ips_auto_ban = anti_abuse_config.max_ips_auto_ban
    
    print(f"🐱 STARTUP: Loaded anti-abuse config - max_ips_per_user: {auth_config.max_ips_per_user}")
    print(f"🐱 STARTUP: Loaded anti-abuse config - max_ips_auto_ban: {auth_config.max_ips_auto_ban}")
except Exception as e:
    print(f"🚫 STARTUP: Failed to load anti-abuse config: {e}")
    print(f"🐱 STARTUP: Using defaults - max_ips_per_user: {auth_config.max_ips_per_user}")

# Run initial health checks in background
key_manager.run_initial_health_checks()

# Request logging middleware removed for cleaner output

@app.route('/health', methods=['GET'])
def health_check():
    start_time = time.time()
    result = jsonify({"status": "healthy", "service": "AI Proxy"})
    metrics.track_request('health', time.time() - start_time)
    return result

@app.route('/openai/v1/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])  # Legacy support
@require_auth
def openai_chat_completions():
    """Proxy OpenAI chat completions with rate limit handling"""
    start_time = time.time()
    max_retries = 3
    
    # Track IP for current prompter count
    client_ip = get_client_ip()
    metrics.track_ip(client_ip)
    
    # Check quota for OpenAI models
    has_quota, quota_error = check_quota('openai')
    if not has_quota:
        return jsonify({"error": quota_error}), 429
    
    # Validate model is whitelisted
    request_json = request.get_json() if request else {}
    model = request_json.get('model', 'gpt-3.5-turbo')
    if not model_manager.is_model_whitelisted(AIProvider.OPENAI, model):
        metrics.track_request('chat_completions', time.time() - start_time, error=True)
        return jsonify({"error": {"message": f"Model '{model}' is not whitelisted for use", "type": "model_not_allowed"}}), 403
    
    # Validate token limits
    model_info = model_manager.get_model_info(model)
    if model_info and (model_info.max_input_tokens or model_info.max_output_tokens):
        print(f"🔍 Token validation for model {model}: max_input={model_info.max_input_tokens}, max_output={model_info.max_output_tokens}")
        
        # Count input tokens using the full request for more accurate counting
        input_tokens = 0
        try:
            token_result = unified_tokenizer.count_tokens(request_json, 'openai', model)
            print(f"🔍 Raw unified tokenizer result: {token_result}")
            # Check for both possible key names
            input_tokens = token_result.get('input_tokens', 0) or token_result.get('prompt_tokens', 0)
            print(f"📊 Unified tokenizer result: {input_tokens} tokens")
        except Exception as token_error:
            print(f"⚠️ Unified tokenizer failed: {token_error}")
            import traceback
            traceback.print_exc()
        
        # If unified tokenizer returned 0, use basic fallback
        if input_tokens == 0:
            print(f"⚠️ Unified tokenizer returned 0, using character estimation...")
            messages = request_json.get('messages', [])
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            input_tokens = max(1, total_chars // 4)  # Conservative 4 chars per token
            print(f"📊 Character fallback: {input_tokens} tokens (from {total_chars} chars)")
        
        print(f"🎯 Final input token count: {input_tokens}")
        
        # Check output tokens limit (from max_completion_tokens or max_tokens)
        max_completion_tokens = request_json.get('max_completion_tokens') or request_json.get('max_tokens')
        print(f"📤 Requested output tokens: {max_completion_tokens}")
        
        # Validate token limits
        is_valid, error_message = model_info.validate_token_limits(input_tokens, max_completion_tokens)
        print(f"✅ Validation result: {is_valid}, message: {error_message}")
        
        if not is_valid:
            print(f"❌ Token limit exceeded, returning error")
            
            # Log token limit violation
            if hasattr(g, 'auth_data') and g.auth_data.get("type") == "user_token":
                user_token = g.auth_data["token"]
                user = g.auth_data["user"]
                ip_hash = hashlib.sha256(g.auth_data["ip"].encode()).hexdigest()
                user_agent = request.headers.get('User-Agent', '')
                
                # Record token violation for happiness tracking
                user.record_token_violation()
                
                # Log the violation event
                structured_logger.log_user_action(
                    user_token=user_token,
                    action='token_limit_exceeded',
                    details={
                        'model': model,
                        'input_tokens': input_tokens,
                        'max_input_tokens': model_info.max_input_tokens,
                        'requested_output_tokens': max_completion_tokens,
                        'max_output_tokens': model_info.max_output_tokens,
                        'error_message': error_message,
                        'ip_hash': ip_hash,
                        'user_agent': user_agent
                    }
                )
            
            metrics.track_request('chat_completions', time.time() - start_time, error=True)
            return jsonify({"error": {"message": error_message, "type": "token_limit_exceeded"}}), 400
    
    for attempt in range(max_retries):
        api_key = key_manager.get_api_key('openai')
        if not api_key:
            metrics.track_request('chat_completions', time.time() - start_time, error=True)
            return jsonify({"error": "No OpenAI API key configured"}), 500
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            session = connection_pool.get_session('openai')
            response = session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=request.json,
                stream=request.json.get('stream', False),
                timeout=getattr(session, '_default_timeout', (10, 30))
            )
            
            response_time = time.time() - start_time
            
            # Handle different response codes
            if response.status_code >= 400:
                error_text = response.text
                if key_manager.handle_api_error('openai', api_key, error_text, response.status_code):
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief pause before retry
                        continue
            else:
                # Success - update key health
                key_manager.update_key_health(api_key, True)
            
            # Extract token information from response
            tokens = None
            response_content = response.content
            request_json = request.get_json() if request else {}
            is_streaming = request_json.get('stream', False) if request_json else False
            
            if response.status_code == 200:
                model = request_json.get('model', 'gpt-3.5-turbo')
                
                # First, always check if OpenAI provided usage data (like Anthropic does)
                try:
                    response_data = json.loads(response_content)
                    if 'usage' in response_data:
                        tokens = response_data['usage']
                        print(f"🐱 Token tracking for {model}: input={tokens.get('prompt_tokens', 0)}, output={tokens.get('completion_tokens', 0)} (API provided)")
                        # Skip all the complex streaming logic if we have usage data
                    elif is_streaming:
                        # Only do complex streaming parsing if no usage data is available
                        pass
                except (json.JSONDecodeError, Exception):
                    tokens = None
                
                # For streaming requests without usage data, count prompt tokens and estimate completion tokens from stream
                if is_streaming and tokens is None:
                    try:
                        # Count prompt tokens
                        token_result = unified_tokenizer.count_tokens(
                            request_data=request_json,
                            service="openai",
                            model=model
                        )
                        
                        # For streaming, we'll estimate completion tokens from the response content
                        completion_tokens = 0
                        try:
                            # Parse streamed response to extract text content
                            response_text = ""
                            if response_content:
                                # For streaming responses, content may be in different formats
                                # Try to extract text from Server-Sent Events (SSE) format
                                content_str = response_content.decode('utf-8') if isinstance(response_content, bytes) else str(response_content)
                                
                                # Parse SSE data chunks - try multiple patterns
                                import re
                                
                                # Pattern 1: Standard SSE format with data: prefix
                                data_chunks = re.findall(r'data: ({.*?})', content_str, re.DOTALL)
                                if not data_chunks:
                                    # Pattern 2: Direct JSON objects (some APIs return this)
                                    data_chunks = re.findall(r'({.*?"choices".*?})', content_str, re.DOTALL)
                                if not data_chunks:
                                    # Pattern 3: Split by newlines and find JSON-like strings
                                    lines = content_str.split('\n')
                                    data_chunks = [line.strip() for line in lines if line.strip().startswith('{') and 'choices' in line]
                                
                                chunks_with_content = 0
                                chunks_processed = 0
                                chunks_parsed_successfully = 0
                                
                                for chunk in data_chunks:
                                    chunks_processed += 1
                                    try:
                                        # Clean up the chunk if it has SSE prefixes
                                        clean_chunk = chunk.strip()
                                        if clean_chunk.startswith('data: '):
                                            clean_chunk = clean_chunk[6:].strip()
                                        
                                        # Skip empty chunks or end markers
                                        if not clean_chunk or clean_chunk == '[DONE]':
                                            continue
                                        
                                        # Try to fix common JSON issues
                                        try:
                                            chunk_data = json.loads(clean_chunk)
                                            chunks_parsed_successfully += 1
                                        except json.JSONDecodeError:
                                            # Try to fix truncated JSON by finding the last complete object
                                            try:
                                                # Find the last complete JSON object
                                                bracket_count = 0
                                                last_complete = ""
                                                for i, char in enumerate(clean_chunk):
                                                    if char == '{':
                                                        bracket_count += 1
                                                    elif char == '}':
                                                        bracket_count -= 1
                                                        if bracket_count == 0:
                                                            last_complete = clean_chunk[:i+1]
                                                            break
                                                
                                                if last_complete:
                                                    chunk_data = json.loads(last_complete)
                                                    chunks_parsed_successfully += 1
                                                else:
                                                    continue
                                            except Exception:
                                                continue
                                        
                                        if 'choices' in chunk_data:
                                            for choice in chunk_data['choices']:
                                                if 'delta' in choice and 'content' in choice['delta']:
                                                    content = choice['delta']['content']
                                                    if content:  # Only add non-empty content
                                                        response_text += content
                                                        chunks_with_content += 1
                                                elif 'message' in choice and 'content' in choice['message']:
                                                    # Some formats put content directly in message
                                                    content = choice['message']['content']
                                                    if content:  # Only add non-empty content
                                                        response_text += content
                                                        chunks_with_content += 1
                                    except Exception:
                                        continue
                            
                            # Estimate completion tokens from collected response text
                            if response_text.strip():
                                completion_token_result = unified_tokenizer.count_tokens(
                                    request_data=request_json,  # Use original request data
                                    service="openai",
                                    model=model,
                                    response_text=response_text.strip()
                                )
                                completion_tokens = completion_token_result.get('completion_tokens', 0)
                            
                        except Exception:
                            completion_tokens = 0
                        
                        # Simple fallback: extract complete AI response and use tiktoken directly
                        if completion_tokens == 0 and response_content:
                            try:
                                content_str = response_content.decode('utf-8') if isinstance(response_content, bytes) else str(response_content)
                                
                                # Extract all delta content to build complete AI response
                                import re
                                complete_response = ""
                                
                                # Extract all content from delta fields in streaming chunks
                                content_matches = re.findall(r'"delta":\s*{[^}]*"content":\s*"([^"]*)"', content_str)
                                complete_response = ''.join(content_matches).replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
                                
                                if complete_response.strip():
                                    # Use tiktoken directly on the complete response
                                    try:
                                        import tiktoken
                                        encoding = tiktoken.encoding_for_model(model if model.startswith('gpt') else 'gpt-3.5-turbo')
                                        completion_tokens = len(encoding.encode(complete_response))
                                        print(f"🐱 Token tracking for {model}: input={token_result['prompt_tokens']}, output={completion_tokens} (tiktoken on complete response)")
                                    except Exception:
                                        # Fallback to unified tokenizer
                                        completion_token_result = unified_tokenizer.count_tokens(
                                            request_data=request_json,
                                            service="openai",
                                            model=model,
                                            response_text=complete_response.strip()
                                        )
                                        completion_tokens = completion_token_result.get('completion_tokens', 0)
                                        print(f"🐱 Token tracking for {model}: input={token_result['prompt_tokens']}, output={completion_tokens} (unified tokenizer fallback)")
                                
                                # Final fallback if no content extracted
                                if completion_tokens == 0:
                                    completion_tokens = max(len(content_str) // 20, 1)  # Very conservative estimate
                                    
                            except Exception:
                                completion_tokens = max(len(response_content) // 20, 1)  # Very conservative estimate
                        
                        tokens = {
                            'prompt_tokens': token_result['prompt_tokens'],
                            'completion_tokens': completion_tokens,
                            'total_tokens': token_result['prompt_tokens'] + completion_tokens
                        }
                        
                        print(f"🐱 DEBUG: Streaming request for {model} - estimated {completion_tokens} completion tokens")
                        
                    except Exception as e:
                        print(f"🚫 DEBUG: Error counting tokens for streaming {model}: {e}")
                        pass
                elif tokens is None:
                    # Non-streaming request without usage data - use simple fallback like Anthropic
                    try:
                        if 'response_data' not in locals():
                            response_data = json.loads(response_content)
                        
                        # Extract response text from choices (like Anthropic extracts from content)
                        response_text = ""
                        if 'choices' in response_data:
                            for choice in response_data['choices']:
                                if 'message' in choice and 'content' in choice['message']:
                                    response_text += choice['message']['content'] + " "
                        
                        # Use unified tokenizer (same as Anthropic approach)
                        token_result = unified_tokenizer.count_tokens(
                            request_data=request_json,
                            service="openai",
                            model=model,
                            response_text=response_text.strip() if response_text else None
                        )
                        tokens = {
                            'prompt_tokens': token_result['prompt_tokens'],
                            'completion_tokens': token_result['completion_tokens'],
                            'total_tokens': token_result['total_tokens']
                        }
                        print(f"🐱 Token tracking for {model}: input={tokens['prompt_tokens']}, output={tokens['completion_tokens']} (unified tokenizer)")
                        
                    except Exception:
                        # Minimal fallback - just count input tokens
                        try:
                            token_result = unified_tokenizer.count_tokens(
                                request_data=request_json,
                                service="openai",
                                model=model
                            )
                            tokens = {
                                'prompt_tokens': token_result['prompt_tokens'],
                                'completion_tokens': 0,
                                'total_tokens': token_result['prompt_tokens']
                            }
                        except Exception:
                            tokens = None
            
            metrics.track_request('chat_completions', response_time, error=response.status_code >= 400, tokens=tokens, service='openai')
            
            # Track token usage for authenticated users
            if tokens and hasattr(g, 'auth_data') and g.auth_data.get('type') == 'user_token':
                # Track model usage and get cost first
                model_cost = model_manager.track_model_usage(
                    user_token=g.auth_data['token'],
                    model_id=model,
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    success=response.status_code == 200
                )
                
                # Track with cost information
                track_token_usage(
                    model, 
                    tokens.get('prompt_tokens', 0), 
                    tokens.get('completion_tokens', 0), 
                    cost=model_cost or 0.0, 
                    response_time_ms=response_time * 1000
                )
                
                # Log completion event
                event_logger.log_chat_completion(
                    token=g.auth_data['token'],
                    model_family='openai',
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    ip_hash=g.auth_data.get('ip', '')
                )
                
                # Log structured completion event with model details
                structured_logger.log_chat_completion(
                    user_token=g.auth_data['token'],
                    model_family='openai',
                    model_name=model,
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    cost_usd=model_cost or 0.0,
                    response_time_ms=response_time * 1000,
                    success=response.status_code == 200,
                    ip_hash=g.auth_data.get('ip', ''),
                    user_agent=request.headers.get('User-Agent')
                )
            
            if is_streaming:
                return Response(
                    response.iter_content(chunk_size=1024),
                    content_type=response.headers.get('content-type'),
                    status=response.status_code
                )
            else:
                return Response(
                    response_content,
                    content_type=response.headers.get('content-type'),
                    status=response.status_code
                )
        
        except Exception as e:
            error_msg = str(e)
            if key_manager.handle_api_error('openai', api_key, error_msg):
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            metrics.track_request('chat_completions', time.time() - start_time, error=True)
            return jsonify({"error": error_msg}), 500
    
    # If we get here, all retries failed
    metrics.track_request('chat_completions', time.time() - start_time, error=True)
    return jsonify({"error": "All API keys rate limited or failed"}), 429

@app.route('/openai/v1/models', methods=['GET'])
@app.route('/v1/models', methods=['GET'])  # Legacy support
def openai_models():
    """Return whitelisted OpenAI models instead of proxying to API"""
    start_time = time.time()
    
    try:
        # Get whitelisted OpenAI models
        whitelisted_models = model_manager.get_whitelisted_models(AIProvider.OPENAI)
        
        # Convert to OpenAI API format
        models_data = []
        for model_info in whitelisted_models:
            models_data.append({
                "id": model_info.model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "openai",
                "permission": [
                    {
                        "id": f"modelperm-{model_info.model_id}",
                        "object": "model_permission",
                        "created": int(datetime.now().timestamp()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": model_info.model_id,
                "parent": None
            })
        
        response_data = {
            "object": "list",
            "data": models_data
        }
        
        metrics.track_request('models', time.time() - start_time)
        return jsonify(response_data)
        
    except Exception as e:
        metrics.track_request('models', time.time() - start_time, error=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/keys/status', methods=['GET'])
def key_status():
    """Check which API keys are configured"""
    start_time = time.time()
    status = {}
    for service, keys in key_manager.api_keys.items():
        status[service] = {
            'configured': len([k for k in keys if k]) > 0,
            'count': len([k for k in keys if k])
        }
    metrics.track_request('key_status', time.time() - start_time)
    return jsonify(status)

@app.route('/api/keys/health', methods=['GET'])
def key_health():
    """Get detailed health status for all keys"""
    start_time = time.time()
    
    # Get health data with safe serialization
    health_data = {}
    with key_manager.lock:
        for key, health in key_manager.key_health.items():
            # Mask the key for security (show only first 8 chars)
            masked_key = key[:8] + '...' if len(key) > 8 else key
            health_data[masked_key] = {
                'status': health['status'],
                'service': health['service'],
                'last_success': health['last_success'].isoformat() if health['last_success'] else None,
                'last_failure': health['last_failure'].isoformat() if health['last_failure'] else None,
                'failure_count': health['failure_count'],
                'consecutive_failures': health['consecutive_failures'],
                'error_type': health['error_type'],
                'last_error': health['last_error'][:100] if health['last_error'] else None  # Truncate error message
            }
    
    metrics.track_request('key_health', time.time() - start_time)
    return jsonify(health_data)

@app.route('/api/debug/keys', methods=['GET'])
def debug_keys():
    """Debug endpoint to check key loading"""
    debug_info = {}
    for service, keys in key_manager.api_keys.items():
        debug_info[service] = {
            'total_keys': len(keys),
            'valid_keys': len([k for k in keys if k]),
            'first_few_chars': [k[:8] + '...' if len(k) > 8 else k for k in keys[:3] if k]  # Show first 3 for debugging
        }
    return jsonify(debug_info)


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get proxy metrics"""
    return jsonify(metrics.get_metrics())

@app.route('/api/system/health', methods=['GET'])
def system_health():
    """Comprehensive system health endpoint"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': metrics.get_uptime(),
            'metrics': metrics.get_metrics(),
            'memory_warning': False,
            'thread_warning': False,
            'recommendations': []
        }
        
        system_stats = health_data['metrics']['system_stats']
        thread_stats = health_data['metrics']['thread_stats']
        
        # Check for warnings
        if system_stats['memory_mb'] > 400:
            health_data['memory_warning'] = True
            health_data['recommendations'].append('Consider reducing memory usage or restarting')
            
        if system_stats['thread_count'] > 50:
            health_data['thread_warning'] = True
            health_data['recommendations'].append('High thread count detected - monitor for thread leaks')
        
        # Check for restarts
        if health_data['metrics']['restart_count'] > 5:
            health_data['recommendations'].append('Frequent restarts detected - investigate stability issues')
        
        # Force garbage collection if needed
        memory_check, collected = metrics.check_memory_threshold(400)
        if memory_check:
            health_data['recommendations'].append(f'Forced garbage collection - {collected} objects collected')
        
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/api/system/gc', methods=['POST'])
def force_gc():
    """Force garbage collection endpoint"""
    try:
        collected = metrics.force_garbage_collection()
        return jsonify({
            'success': True,
            'objects_collected': collected,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Old admin dashboard route removed - now handled by admin_web_bp

@app.route('/', methods=['GET'])
def dashboard():
    """Dashboard webpage"""
    metrics_data = metrics.get_metrics()
    key_status_data = {}
    key_health_data = {}
    
    for service, keys in key_manager.api_keys.items():
        key_status_data[service] = {
            'configured': len([k for k in keys if k and k.strip()]) > 0,
            'count': len([k for k in keys if k and k.strip()])
        }
        
        
        # Get health status for this service
        service_health = {
            'healthy': 0,
            'rate_limited': 0,
            'invalid_key': 0,
            'quota_exceeded': 0,
            'unknown': 0,
            'failed': 0,
            'key_details': [],
            'successful_requests': 0,
            'total_tokens': 0,
            'avg_response_time': '0.00'
        }
        
        # Get metrics data for this service
        if service in metrics_data['service_tokens']:
            service_metrics = metrics_data['service_tokens'][service]
            service_health['successful_requests'] = service_metrics['successful_requests']
            service_health['total_tokens'] = service_metrics['total_tokens']
            
            # Calculate average response time for this service
            if service_metrics['response_times']:
                avg_time = sum(service_metrics['response_times']) / len(service_metrics['response_times'])
                service_health['avg_response_time'] = f"{avg_time:.2f}"
        
        with key_manager.lock:
            for key in keys:
                if key and key in key_manager.key_health:
                    health = key_manager.key_health[key]
                    status = health['status']
                    
                    
                    # Count by status
                    if status in service_health:
                        service_health[status] += 1
                    else:
                        service_health['failed'] += 1
                    
                    # Add key details (masked for security)
                    masked_key = key[:8] + '...' if len(key) > 8 else key
                    service_health['key_details'].append({
                        'key': masked_key,
                        'status': status,
                        'last_success': health['last_success'].strftime('%Y-%m-%d %H:%M:%S') if health['last_success'] else 'Never',
                        'last_failure': health['last_failure'].strftime('%Y-%m-%d %H:%M:%S') if health['last_failure'] else 'Never',
                        'consecutive_failures': health['consecutive_failures'],
                        'error_type': health['error_type']
                    })
                else:
                    service_health['unknown'] += 1
        
        key_health_data[service] = service_health
    
    uptime_hours = metrics_data['uptime_seconds'] / 3600
    uptime_display = f"{uptime_hours:.1f} hours" if uptime_hours >= 1 else f"{metrics_data['uptime_seconds']:.0f} seconds"
    
    # Get current deployment URL
    base_url = get_base_url()
    
    # Get configurable dashboard variables from environment
    dashboard_config = {
        'title': f"{os.getenv('BRAND_EMOJI', '🐱')} {os.getenv('DASHBOARD_TITLE', 'NyanProxy Dashboard')}",
        'subtitle': os.getenv('DASHBOARD_SUBTITLE', 'Purr-fect monitoring for your AI service proxy!'),
        'refresh_interval': int(os.getenv('DASHBOARD_REFRESH_INTERVAL', 30)),
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
        'brand_description': os.getenv('BRAND_DESCRIPTION', 'Meow-nificent AI Proxy!'),
        'auth_mode': auth_config.mode,
        'auth_display': {
            'none': 'No Authentication',
            'proxy_key': 'Proxy Key (Shared Password)', 
            'user_token': 'User Token (Individual Authentication)'
        }.get(auth_config.mode, 'Unknown')
    }
    
    return render_template('dashboard.html',
                         base_url=base_url,
                         uptime_display=uptime_display,
                         total_requests=metrics_data['total_requests'],
                         total_prompts=metrics_data['request_counts']['chat_completions'],
                         total_tokens=metrics_data['total_tokens'],
                         average_response_time=f"{metrics_data['average_response_time']:.2f}s",
                         current_prompters=metrics_data['current_prompters'],
                         key_status_data=key_status_data,
                         key_health_data=key_health_data,
                         request_counts=metrics_data['request_counts'],
                         error_counts=metrics_data['error_counts'],
                         last_request=metrics_data['last_request'],
                         config=dashboard_config)

def get_base_url():
    """Get the base URL of the current deployment"""
    # Try to detect various deployment platforms automatically
    
    # Hugging Face Spaces - check for HF_SPACE_ID or detect from hostname
    if os.getenv('HF_SPACE_ID'):
        space_id = os.getenv('HF_SPACE_ID')
        space_url = space_id.replace('/', '-')
        return f'https://{space_url}.hf.space'
    
    # Render.com
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    if render_url:
        return render_url
    
    # Railway
    railway_url = os.getenv('RAILWAY_STATIC_URL')
    if railway_url:
        return railway_url
    
    # Heroku
    heroku_app = os.getenv('HEROKU_APP_NAME')
    if heroku_app:
        return f'https://{heroku_app}.herokuapp.com'
    
    # Vercel
    vercel_url = os.getenv('VERCEL_URL')
    if vercel_url:
        return f'https://{vercel_url}'
    
    # Netlify
    netlify_url = os.getenv('NETLIFY_URL')
    if netlify_url:
        return netlify_url
    
    # Generic detection from common environment variables
    app_url = os.getenv('APP_URL')
    if app_url:
        return app_url
    
    # Try to detect from request headers (if available)
    try:
        if request and request.headers:
            host = request.headers.get('Host')
            proto = 'https' if request.headers.get('X-Forwarded-Proto') == 'https' else 'http'
            if host and 'localhost' not in host:
                return f'{proto}://{host}'
    except:
        pass
    
    # Fallback to localhost for development
    port = os.getenv('PORT', 7860)
    return f'http://localhost:{port}'

# Add Anthropic endpoints
@app.route('/anthropic/v1/messages', methods=['POST'])
@app.route('/v1/messages', methods=['POST'])  # Legacy support
@app.route('/anthropic/v1', methods=['POST'])  # User-friendly endpoint
@require_auth
def anthropic_messages():
    """Proxy Anthropic messages with token tracking"""
    # Anthropic endpoint called
    start_time = time.time()
    max_retries = 3
    
    # Track IP for current prompter count
    client_ip = get_client_ip()
    metrics.track_ip(client_ip)
    
    # Check quota for Anthropic models
    has_quota, quota_error = check_quota('anthropic')
    if not has_quota:
        return jsonify({"error": quota_error}), 429
    
    # Validate model is whitelisted
    request_json = request.get_json() if request else {}
    model = request_json.get('model', 'claude-3-haiku-20240307')
    if not model_manager.is_model_whitelisted(AIProvider.ANTHROPIC, model):
        metrics.track_request('chat_completions', time.time() - start_time, error=True)
        return jsonify({"error": {"message": f"Model '{model}' is not whitelisted for use", "type": "model_not_allowed"}}), 403
    
    # Validate token limits
    model_info = model_manager.get_model_info(model)
    if model_info and (model_info.max_input_tokens or model_info.max_output_tokens):
        # Count input tokens using the full request for more accurate counting
        input_tokens = 0
        try:
            token_result = unified_tokenizer.count_tokens(request_json, 'anthropic')
            print(f"🔍 Raw unified tokenizer result (Anthropic): {token_result}")
            # Check for both possible key names
            input_tokens = token_result.get('input_tokens', 0) or token_result.get('prompt_tokens', 0)
            print(f"📊 Unified tokenizer result: {input_tokens} tokens")
        except Exception as token_error:
            print(f"⚠️ Unified tokenizer failed: {token_error}")
            # Fallback to character estimation for Anthropic
            messages = request_json.get('messages', [])
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            input_tokens = max(1, total_chars // 4)
            print(f"📊 Character fallback for Anthropic: {input_tokens} tokens")
        
        # Check output tokens limit (from max_tokens)
        max_completion_tokens = request_json.get('max_tokens')
        
        # Validate token limits
        is_valid, error_message = model_info.validate_token_limits(input_tokens, max_completion_tokens)
        if not is_valid:
            metrics.track_request('chat_completions', time.time() - start_time, error=True)
            return jsonify({"error": {"message": error_message, "type": "token_limit_exceeded"}}), 400
    
    for attempt in range(max_retries):
        api_key = key_manager.get_api_key('anthropic')
        if not api_key:
            metrics.track_request('chat_completions', time.time() - start_time, error=True)
            return jsonify({"error": "No Anthropic API key configured"}), 500
        
        headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01'
        }
        
        try:
            session = connection_pool.get_session('anthropic')
            response = session.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=request.json,
                stream=request.json.get('stream', False),
                timeout=getattr(session, '_default_timeout', (10, 30))
            )
            
            response_time = time.time() - start_time
            
            # Handle different response codes
            if response.status_code >= 400:
                error_text = response.text
                if key_manager.handle_api_error('anthropic', api_key, error_text, response.status_code):
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief pause before retry
                        continue
            else:
                # Success - update key health
                key_manager.update_key_health(api_key, True)
            
            # Extract token information from response
            tokens = None
            response_content = response.content
            
            request_json = request.get_json() if request else {}
            is_streaming = request_json.get('stream', False) if request_json else False
            
            # Anthropic response received
            
            # Debug: Print raw response to see structure
            if response.status_code == 200:
                try:
                    debug_response = json.loads(response_content)
                    # Anthropic response parsed
                except:
                    # Could not parse response as JSON
                    pass
            
            if response.status_code == 200:
                # Always try to count tokens, regardless of streaming
                try:
                    response_data = json.loads(response_content)
                    
                    # First try to get tokens from API response
                    if 'usage' in response_data:
                        usage = response_data['usage']
                        tokens = {
                            'prompt_tokens': usage.get('input_tokens', 0),
                            'completion_tokens': usage.get('output_tokens', 0),
                            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                        }
                    else:
                        # Fallback: use tokenizer to estimate
                        response_text = ""
                        if 'content' in response_data:
                            for content in response_data['content']:
                                if content.get('type') == 'text':
                                    response_text += content.get('text', '') + " "
                        
                        # Use tokenizer to count tokens
                        token_result = unified_tokenizer.count_tokens(
                            request_data=request_json,
                            service="anthropic",
                            response_text=response_text.strip() if response_text else None
                        )
                        tokens = {
                            'prompt_tokens': token_result['prompt_tokens'],
                            'completion_tokens': token_result['completion_tokens'],
                            'total_tokens': token_result['total_tokens']
                        }
                        
                except Exception as e:
                    # Fallback: count request tokens at minimum
                    try:
                        # Extract text from request for token counting
                        request_text = ""
                        if 'messages' in request_json:
                            for msg in request_json['messages']:
                                if 'content' in msg:
                                    request_text += str(msg['content']) + " "
                        if 'system' in request_json:
                            request_text += str(request_json['system']) + " "
                        
                        # Use simple character-based estimation if tokenizer fails
                        estimated_prompt_tokens = max(len(request_text) // 4, 100)  # ~4 chars per token
                        estimated_completion_tokens = 50  # Default completion estimate
                        
                        tokens = {
                            'prompt_tokens': estimated_prompt_tokens,
                            'completion_tokens': estimated_completion_tokens,
                            'total_tokens': estimated_prompt_tokens + estimated_completion_tokens
                        }
                    except Exception:
                        # Absolute fallback
                        tokens = {
                            'prompt_tokens': 1000,  # Higher default for large requests
                            'completion_tokens': 100,
                            'total_tokens': 1100
                        }
            
            # Tracking Anthropic request
            metrics.track_request('chat_completions', response_time, error=response.status_code >= 400, tokens=tokens, service='anthropic')
            
            # Track token usage for authenticated users
            if tokens and hasattr(g, 'auth_data') and g.auth_data.get('type') == 'user_token':
                # Track model usage and get cost first
                model_cost = model_manager.track_model_usage(
                    user_token=g.auth_data['token'],
                    model_id=model,
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    success=response.status_code == 200
                )
                
                # Track with cost information
                track_token_usage(
                    model, 
                    tokens.get('prompt_tokens', 0), 
                    tokens.get('completion_tokens', 0), 
                    cost=model_cost or 0.0, 
                    response_time_ms=response_time * 1000
                )
                
                # Log completion event
                event_logger.log_chat_completion(
                    token=g.auth_data['token'],
                    model_family='anthropic',
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    ip_hash=g.auth_data.get('ip', '')
                )
            
            if is_streaming:
                return Response(
                    response.iter_content(chunk_size=1024),
                    content_type=response.headers.get('content-type'),
                    status=response.status_code
                )
            else:
                return Response(
                    response_content,
                    content_type=response.headers.get('content-type'),
                    status=response.status_code
                )
        
        except Exception as e:
            error_msg = str(e)
            if key_manager.handle_api_error('anthropic', api_key, error_msg):
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
            
            metrics.track_request('chat_completions', time.time() - start_time, error=True)
            return jsonify({"error": error_msg}), 500
    
    # If we get here, all retries failed
    metrics.track_request('chat_completions', time.time() - start_time, error=True)
    return jsonify({"error": "All API keys rate limited or failed"}), 429

@app.route('/anthropic/v1/models', methods=['GET'])
def anthropic_models():
    """Anthropic models endpoint (placeholder)"""
    return jsonify({"error": "Anthropic models endpoint not implemented yet"}), 501


# ============================================================================
# 🔮 GOOGLE GEMINI PROXY ENDPOINTS
# ============================================================================

@app.route('/google/v1/models/<path:model_path>', methods=['POST'])
@app.route('/google/v1/<path:model_path>', methods=['POST'])
def google_gemini_proxy(model_path):
    """Google Gemini API proxy endpoint"""
    try:
        # Check authentication
        success, error_message, user_data = authenticate_request()
        if not success:
            return jsonify({"error": {"message": error_message, "type": "auth_error"}}), 401
        
        # Store auth data in Flask g object for use in endpoint
        g.auth_data = user_data
        
        # Get request data
        request_json = request.get_json() or {}
        
        # Extract model name from path for validation
        model_name = model_path.split(':')[0]  # Remove :generateContent if present
        
        # Handle various Google API path formats
        if 'models/' in model_name:
            # Extract everything after the last 'models/'
            model_name = model_name.split('models/')[-1]
        
        # Remove any remaining path prefixes (like v1beta/, v1/, etc.)
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        
        
        # Check if model is whitelisted
        is_whitelisted = model_manager.is_model_whitelisted(AIProvider.GOOGLE, model_name)
        
        if not is_whitelisted:
            return jsonify({"error": {"message": f"Model {model_name} is not whitelisted", "type": "model_not_allowed"}}), 403
        
        # Get Google API key
        selected_key = key_manager.get_api_key('google')
        if not selected_key:
            return jsonify({"error": {"message": "No healthy Google API keys available", "type": "service_unavailable"}}), 503
        
        # Construct Google API URL
        # The model_path contains the full path after /google/v1/
        # We need to construct the correct Google API URL
        if model_path.startswith('v1beta/'):
            # Already has v1beta prefix, use as-is
            google_url = f'https://generativelanguage.googleapis.com/{model_path}'
        elif model_path.startswith('models/'):
            # Add v1beta prefix
            google_url = f'https://generativelanguage.googleapis.com/v1beta/{model_path}'
        else:
            # Assume it's a model name, add full path
            google_url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_path}'
        
        
        # Check if this is a streaming request
        is_streaming = 'streamGenerateContent' in model_path or request.args.get('alt') == 'sse'
        
        # Prepare request body according to old project's approach
        # Remove 'stream' and 'model' from request body before sending to Google
        google_request_body = {k: v for k, v in request_json.items() if k not in ['stream', 'model']}
        
        
        # Forward request to Google
        try:
            if is_streaming:
                # Handle streaming response - Use API key in query string and add alt=sse
                # Modify URL to include API key and alt=sse query parameters
                streaming_url = google_url
                if '?' in streaming_url:
                    streaming_url += f'&key={selected_key}&alt=sse'
                else:
                    streaming_url += f'?key={selected_key}&alt=sse'
                
                
                session = connection_pool.get_session('google')
                response = session.post(
                    streaming_url,
                    json=google_request_body,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'NyanProxy/1.0'
                    },
                    timeout=(10, 120),  # (connect_timeout, read_timeout)
                    stream=True
                )
            else:
                # Handle regular response - Use API key in query string
                regular_url = google_url
                if '?' in regular_url:
                    regular_url += f'&key={selected_key}'
                else:
                    regular_url += f'?key={selected_key}'
                
                
                session = connection_pool.get_session('google')
                response = session.post(
                    regular_url,
                    json=google_request_body,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'NyanProxy/1.0'
                    },
                    timeout=(10, 120)  # (connect_timeout, read_timeout)
                )
            
            
            # Handle response
            if response.status_code == 200:
                if is_streaming:
                    # Handle streaming response with background token counting
                    all_chunks = []  # Store chunks for background token parsing
                    
                    def generate():
                        try:
                            chunk_count = 0
                            for chunk in response.iter_content(chunk_size=8192, decode_unicode=False):
                                if chunk:
                                    chunk_count += 1
                                    
                                    # Store chunk for background token parsing (non-blocking)
                                    all_chunks.append(chunk)
                                    
                                    
                                    # Yield chunk immediately - no processing delays
                                    yield chunk
                            
                                
                        except Exception as e:
                            print(f"❌ Streaming error: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            # Send error as SSE event
                            yield f"data: {{'error': 'Stream interrupted: {str(e)}'}}\n\n"
                    
                    # Start background token counting after response setup
                    def background_token_counting():
                        try:
                            # Wait a moment for streaming to complete
                            time.sleep(1)
                            
                            input_tokens = 0
                            output_tokens = 0
                            
                            # Parse all collected chunks for usage metadata
                            for chunk in all_chunks:
                                try:
                                    chunk_text = chunk.decode('utf-8')
                                    if 'usageMetadata' in chunk_text:
                                        # Parse JSON from SSE data
                                        lines = chunk_text.strip().split('\n')
                                        for line in lines:
                                            if line.startswith('data: ') and line != 'data: [DONE]':
                                                try:
                                                    json_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                                    if 'usageMetadata' in json_data:
                                                        usage = json_data['usageMetadata']
                                                        input_tokens = usage.get('promptTokenCount', 0)
                                                        output_tokens = usage.get('candidatesTokenCount', 0)
                                                        break
                                                except json.JSONDecodeError:
                                                    continue
                                except UnicodeDecodeError:
                                    continue
                            
                            # Track usage if tokens were found
                            if input_tokens > 0 or output_tokens > 0:
                                # Use stored auth data (Flask context is gone in background thread)
                                if auth_data_copy and auth_data_copy.get("type") == "user_token":
                                    user_token = auth_data_copy['token']
                                    
                                    # Track model usage
                                    model_cost = model_manager.track_model_usage(
                                        user_token=user_token,
                                        model_id=model_name,
                                        input_tokens=input_tokens,
                                        output_tokens=output_tokens
                                    )
                                    
                                    # Track user usage
                                    user = user_store.get_user(user_token)
                                    if user:
                                        # Get IP from stored auth data or use placeholder
                                        client_ip = auth_data_copy.get('ip', '127.0.0.1')
                                        user.add_request_tracking(
                                            model_family=model_name,
                                            input_tokens=input_tokens,
                                            output_tokens=output_tokens,
                                            cost=model_cost or 0.0,
                                            ip_hash=client_ip
                                        )
                                
                        except Exception as e:
                            # Background token counting failed silently
                            import traceback
                            traceback.print_exc()
                    
                    # Store auth data for background thread (Flask context will be gone)
                    auth_data_copy = None
                    if hasattr(g, 'auth_data'):
                        auth_data_copy = g.auth_data.copy()
                        # Store client IP for background tracking
                        auth_data_copy['ip'] = get_client_ip()
                    
                    # Start managed background thread
                    thread_manager.create_daemon_thread(
                        background_token_counting, 
                        f"token_counter_google_{model_name}"
                    )
                    
                    # Update key health for successful streaming
                    key_manager.update_key_health(selected_key, True)
                    
                    # Return streaming response with proper SSE headers
                    from flask import Response as FlaskResponse
                    return FlaskResponse(
                        generate(),
                        status=200,
                        headers={
                            'Content-Type': 'text/event-stream; charset=utf-8',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                            'X-Accel-Buffering': 'no',  # nginx-specific fix
                            'Access-Control-Allow-Origin': '*',
                            'Access-Control-Allow-Headers': 'Cache-Control'
                        }
                    )
                else:
                    # Handle regular JSON response
                    response_data = response.json()
                    
                    # Extract usage info for tracking
                    input_tokens = 0
                    output_tokens = 0
                    
                    # Google Gemini response structure
                    if 'usageMetadata' in response_data:
                        usage = response_data['usageMetadata']
                        input_tokens = usage.get('promptTokenCount', 0)
                        output_tokens = usage.get('candidatesTokenCount', 0)
                    
                    # Track model usage and calculate cost
                    model_cost = 0.0
                    if hasattr(g, 'auth_data') and g.auth_data.get("type") == "user_token":
                        user_token = g.auth_data['token']
                        model_cost = model_manager.track_model_usage(
                            user_token=user_token,
                            model_id=model_name,  # Use extracted model name, not full path
                            input_tokens=input_tokens,
                            output_tokens=output_tokens
                        )
                        
                        # Get user object and track usage
                        user = user_store.get_user(user_token)
                        if user:
                            user.add_request_tracking(
                                model_family=model_name,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cost=model_cost or 0.0,
                                ip_hash=get_client_ip()
                            )
                    
                    # Update key health
                    key_manager.update_key_health(selected_key, True)
                    
                    return jsonify(response_data), 200
            
            else:
                # Handle error response
                error_msg = response.text
                key_manager.update_key_health(selected_key, False, response.status_code, error_msg)
                
                try:
                    error_data = response.json()
                    return jsonify(error_data), response.status_code
                except:
                    return jsonify({
                        "error": {
                            "message": f"Google API error: {error_msg}",
                            "type": "google_api_error"
                        }
                    }), response.status_code
                    
        except requests.exceptions.Timeout:
            key_manager.update_key_health(selected_key, False, None, "Request timeout")
            return jsonify({"error": {"message": "Request timeout", "type": "timeout_error"}}), 504
            
        except requests.exceptions.RequestException as e:
            key_manager.update_key_health(selected_key, False, None, str(e))
            return jsonify({"error": {"message": f"Network error: {str(e)}", "type": "network_error"}}), 502
    
    except Exception as e:
        print(f"❌ Google Gemini proxy error: {str(e)}")
        return jsonify({"error": {"message": "Internal server error", "type": "internal_error"}}), 500


@app.route('/google/v1/models', methods=['GET'])
def google_models():
    """Google models endpoint"""
    try:
        # Check authentication
        auth_result, user = authenticate()
        if auth_result != AuthResult.SUCCESS:
            return jsonify({"error": {"message": "Authentication required", "type": "auth_error"}}), 401
        
        # Get Google API key
        selected_key = key_manager.select_key('google')
        if not selected_key:
            return jsonify({"error": {"message": "No Google API keys available", "type": "service_unavailable"}}), 503
        
        # Get models from Google
        try:
            response = requests.get(
                f'https://generativelanguage.googleapis.com/v1/models?key={selected_key}',
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json(), 200
            else:
                return jsonify({"error": {"message": "Failed to fetch Google models", "type": "google_api_error"}}), response.status_code
                
        except requests.exceptions.RequestException as e:
            return jsonify({"error": {"message": f"Network error: {str(e)}", "type": "network_error"}}), 502
    
    except Exception as e:
        print(f"❌ Google models error: {str(e)}")
        return jsonify({"error": {"message": "Internal server error", "type": "internal_error"}}), 500

