#!/usr/bin/env python3
"""
Startup script for NyanProxy
"""
import sys
import os
import signal
import atexit
import time

# Add the project root to Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def shutdown_handler(signum=None, frame=None):
    """Gracefully handle shutdown"""
    print("[CAT] Shutting down NyanProxy...")
    shutdown_start = time.time()
    
    try:
        # Import core components for shutdown
        from core.app import thread_manager, connection_pool, metrics
        
        print("[THREAD] Shutting down thread manager...")
        thread_manager.shutdown_all()
        
        print("[LINK] Closing connection pools...")
        connection_pool.close_all_sessions()
        
        print("[CHART] Final garbage collection...")
        metrics.force_garbage_collection()
        
        print(" Shutting down model manager...")
        from src.services.model_families import model_manager
        model_manager.shutdown()
        
        # Record clean shutdown
        try:
            from src.services.firebase_logger import structured_logger
            structured_logger.log_system_event(
                event_type='shutdown',
                details={
                    'shutdown_duration': time.time() - shutdown_start,
                    'process_id': os.getpid(),
                    'clean_shutdown': True
                }
            )
        except Exception:
            pass  # Don't fail shutdown on logging error
            
        print(f"[OK] Clean shutdown completed in {time.time() - shutdown_start:.2f}s")
        
    except Exception as e:
        print(f"[WARN] Error during shutdown: {e}")
        # Still attempt to record the shutdown attempt
        try:
            from src.services.firebase_logger import structured_logger
            structured_logger.log_system_event(
                event_type='shutdown',
                details={
                    'shutdown_duration': time.time() - shutdown_start,
                    'process_id': os.getpid(),
                    'clean_shutdown': False,
                    'error': str(e)
                }
            )
        except Exception:
            pass
    
    if signum is not None:
        sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
atexit.register(shutdown_handler)

# Run the application
if __name__ == '__main__':
    from core.app import app
    
    port = int(os.getenv('PORT', 7860))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    
    if debug_mode:
        # Development mode with debugging
        app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
    else:
        # Production mode with threading and better performance
        app.run(
            host='0.0.0.0', 
            port=port, 
            debug=False, 
            threaded=True,
            processes=1,
            use_reloader=False
        )