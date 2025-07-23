"""
Flask Compatibility Layer for FastAPI Migration

Provides Flask-compatible request objects and global variables
to ensure our extracted handlers work seamlessly with FastAPI
"""
from typing import Dict, Any, Optional
from fastapi import Request
import json


class MockFlaskRequest:
    """Mock Flask request object for FastAPI compatibility"""
    
    def __init__(self, fastapi_request: Request, json_data: Dict[str, Any] = None):
        self.fastapi_request = fastapi_request
        self._json_data = json_data
        self.headers = fastapi_request.headers
        self.method = fastapi_request.method
        self.url = str(fastapi_request.url)
        self.remote_addr = fastapi_request.client.host if fastapi_request.client else "unknown"
        
    def get_json(self, **kwargs):
        """Get JSON data from request"""
        return self._json_data or {}
    
    @property
    def json(self):
        """JSON property for compatibility"""
        return self._json_data or {}
    
    def get_data(self, **kwargs):
        """Get raw request data"""
        if self._json_data:
            return json.dumps(self._json_data).encode('utf-8')
        return b'{}'


class MockFlaskG:
    """Mock Flask g object for request-scoped data"""
    
    def __init__(self, auth_data: Dict[str, Any] = None):
        self.auth_data = auth_data or {}


class FlaskCompatibilityContext:
    """Context manager to provide Flask compatibility during handler execution"""
    
    def __init__(self, fastapi_request: Request, json_data: Dict[str, Any] = None, auth_data: Dict[str, Any] = None):
        self.mock_request = MockFlaskRequest(fastapi_request, json_data)
        self.mock_g = MockFlaskG(auth_data)
        self.original_request = None
        self.original_g = None
        
    def __enter__(self):
        """Set up Flask-compatible globals"""
        import sys
        
        # Store original values if they exist
        if 'flask' in sys.modules:
            flask_module = sys.modules['flask']
            if hasattr(flask_module, 'request'):
                self.original_request = flask_module.request
            if hasattr(flask_module, 'g'):
                self.original_g = flask_module.g
            
            # Replace with our mocks
            flask_module.request = self.mock_request
            flask_module.g = self.mock_g
        
        # Also make them available globally for imports
        import builtins
        builtins.request = self.mock_request
        builtins.g = self.mock_g
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original Flask globals"""
        import sys
        
        # Restore Flask module globals if they existed
        if 'flask' in sys.modules:
            flask_module = sys.modules['flask']
            if self.original_request is not None:
                flask_module.request = self.original_request
            elif hasattr(flask_module, 'request'):
                delattr(flask_module, 'request')
                
            if self.original_g is not None:
                flask_module.g = self.original_g
            elif hasattr(flask_module, 'g'):
                delattr(flask_module, 'g')
        
        # Clean up builtins
        import builtins
        if hasattr(builtins, 'request'):
            delattr(builtins, 'request')
        if hasattr(builtins, 'g'):
            delattr(builtins, 'g')


def convert_flask_response(flask_result, status_code: int):
    """Convert Flask response to FastAPI response"""
    try:
        # Handle different Flask response types
        if hasattr(flask_result, 'get_json'):
            # Flask jsonify response
            return flask_result.get_json()
        elif hasattr(flask_result, 'json'):
            # Response with json attribute
            return flask_result.json
        elif hasattr(flask_result, 'data'):
            # Raw response data
            import json
            try:
                return json.loads(flask_result.data.decode('utf-8'))
            except:
                return {"data": flask_result.data.decode('utf-8')}
        elif isinstance(flask_result, dict):
            # Already a dictionary
            return flask_result
        else:
            # Convert to string representation
            return {"response": str(flask_result)}
    except Exception as e:
        # Fallback for any conversion issues
        return {"error": f"Response conversion failed: {str(e)}"}


def make_flask_compatible(handler_method):
    """Decorator to make handler methods Flask-compatible"""
    def wrapper(handler_instance, fastapi_request: Request, json_data: Dict[str, Any] = None, auth_data: Dict[str, Any] = None):
        with FlaskCompatibilityContext(fastapi_request, json_data, auth_data):
            return handler_method(handler_instance)
    return wrapper