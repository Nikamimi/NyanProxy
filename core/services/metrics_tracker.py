"""
Metrics Tracking Service for NyanProxy

Comprehensive metrics and system monitoring
Extracted from core/app.py for better modularity
"""
import time
import threading
import gc
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from .thread_manager import ThreadManager


class MetricsTracker:
    """Comprehensive metrics and system monitoring"""
    
    def __init__(self, thread_manager: ThreadManager = None):
        self.start_time = time.time()
        self.process = psutil.Process()  # Current process for monitoring
        self.thread_manager = thread_manager  # Dependency injection
        
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
    
    def track_request(self, endpoint: str, response_time: float = None, error: bool = False, 
                     tokens: dict = None, service: str = None):
        """Track request metrics with comprehensive logging"""
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
                
                # Track successful requests (only count if not an error)
                if not error:
                    self.service_tokens[service]['successful_requests'] += 1
                
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
        """Get uptime in seconds"""
        return time.time() - self.start_time
    
    def get_average_response_time(self):
        """Calculate average response time from recent responses"""
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
            print(f"[WARN] Error getting system stats: {e}")
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
            print(f"[CLEAN] Garbage collection: {collected} objects collected")
            return collected
        except Exception as e:
            print(f"[WARN] Error during garbage collection: {e}")
            return 0
    
    def check_memory_threshold(self, threshold_mb=500):
        """Check if memory usage exceeds threshold and force GC if needed"""
        try:
            current_memory = self.process.memory_info().rss / (1024 * 1024)
            if current_memory > threshold_mb:
                print(f" Memory usage high: {current_memory:.2f}MB, forcing garbage collection")
                collected = self.force_garbage_collection()
                return True, collected
            return False, 0
        except Exception as e:
            print(f"[WARN] Error checking memory threshold: {e}")
            return False, 0
    
    def get_metrics(self):
        """Get comprehensive metrics data"""
        with self.lock:
            system_stats = self.get_system_stats()
            
            # Get thread stats from thread manager if available
            thread_stats = {}
            if self.thread_manager:
                thread_stats = self.thread_manager.get_thread_stats()
                # Periodic cleanup of dead threads
                self.thread_manager.cleanup_dead_threads()
            
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