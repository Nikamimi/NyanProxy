"""
Thread Management Service for NyanProxy

Manages background threads and prevents resource leaks
Extracted from core/app.py for better modularity
"""
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any


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
            print(f"Created thread pool '{pool_name}' with {max_workers} workers")
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
                print(f"Thread {thread_id} error: {e}")
            finally:
                with self.lock:
                    if thread_id in self.active_threads:
                        del self.active_threads[thread_id]
                        print(f"Cleaned up thread {thread_id}")
        
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
        print(f"Started managed thread: {thread_id}")
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
                print(f"Cleaned up {len(dead_threads)} dead threads")
    
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
        print("Shutting down thread manager...")
        self.shutdown_event.set()
        
        # Shutdown thread pools
        for pool_name, pool in self.executor_pools.items():
            print(f"Shutting down pool: {pool_name}")
            pool.shutdown(wait=True, cancel_futures=True)
        
        print("Thread manager shutdown complete")