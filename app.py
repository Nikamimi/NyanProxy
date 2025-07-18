#!/usr/bin/env python3
"""
Main application entry point for Hugging Face compatibility
"""
import sys
import os

# Add the project root to Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app from the core module
from core.app import app

if __name__ == '__main__':
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