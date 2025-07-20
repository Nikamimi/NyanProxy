#!/usr/bin/env python3
"""
Startup script for NyanProxy
"""
import sys
import os
import signal
import atexit

# Add the project root to Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def shutdown_handler(signum=None, frame=None):
    """Gracefully handle shutdown"""
    print("🐱 Shutting down NyanProxy...")
    try:
        from src.services.model_families import model_manager
        model_manager.shutdown()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    
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