"""
Base Handler Class for NyanProxy

Provides common functionality for all route handlers
Dependency injection pattern for services
"""
from abc import ABC
from typing import Any, Dict
from flask import request, jsonify, g
import time
import hashlib


class BaseHandler(ABC):
    """Base class for all route handlers with common functionality"""
    
    def __init__(self, services: Dict[str, Any]):
        """Initialize handler with injected services
        
        Args:
            services: Dictionary containing all required services
                - thread_manager: ThreadManager instance
                - connection_pool: ConnectionPoolManager instance  
                - metrics: MetricsTracker instance
                - key_manager: API key manager instance
                - model_manager: Model manager instance
                - user_store: User store instance
                - event_logger: Event logger instance
                - structured_logger: Structured logger instance
        """
        self.thread_manager = services['thread_manager']
        self.connection_pool = services['connection_pool']
        self.metrics = services['metrics']
        self.key_manager = services['key_manager']
        self.model_manager = services['model_manager']
        self.user_store = services['user_store']
        self.event_logger = services['event_logger']
        self.structured_logger = services['structured_logger']
    
    def get_client_ip(self) -> str:
        """Get client IP address from request"""
        # This should match the logic from src/middleware/auth.py
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.headers.get('X-Real-IP', request.remote_addr)
    
    def track_request_start(self) -> float:
        """Mark the start of request processing"""
        start_time = time.time()
        
        # Track IP for current prompter count
        client_ip = self.get_client_ip()
        self.metrics.track_ip(client_ip)
        
        return start_time
    
    def track_request_end(self, endpoint: str, start_time: float, error: bool = False, 
                         tokens: Dict = None, service: str = None):
        """Track request completion"""
        response_time = time.time() - start_time
        self.metrics.track_request(endpoint, response_time, error=error, tokens=tokens, service=service)
    
    def get_auth_data(self) -> Dict[str, Any]:
        """Get authentication data from Flask g object"""
        return getattr(g, 'auth_data', {})
    
    def create_error_response(self, message: str, status_code: int = 500) -> tuple:
        """Create standardized error response"""
        return jsonify({"error": message}), status_code
    
    def create_success_response(self, data: Any) -> Any:
        """Create standardized success response"""
        if isinstance(data, dict):
            return jsonify(data)
        return data
    
    def log_completion_event(self, model_family: str, model_name: str, tokens: Dict[str, int], 
                           cost: float, response_time: float, success: bool):
        """Log completion event with all loggers"""
        auth_data = self.get_auth_data()
        if auth_data.get('type') == 'user_token':
            # Track model usage and get cost
            if hasattr(self.model_manager, 'track_model_usage'):
                model_cost = self.model_manager.track_model_usage(
                    user_token=auth_data['token'],
                    model_id=model_name,
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    success=success
                )
            
            # Log with event logger
            if hasattr(self.event_logger, 'log_chat_completion'):
                self.event_logger.log_chat_completion(
                    token=auth_data['token'],
                    model_family=model_family,
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    ip_hash=auth_data.get('ip', '')
                )
            
            # Log with structured logger
            if hasattr(self.structured_logger, 'log_chat_completion'):
                client_ip = self.get_client_ip()
                ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
                
                self.structured_logger.log_chat_completion(
                    user_token=auth_data['token'],
                    model_family=model_family,
                    model_name=model_name,
                    input_tokens=tokens.get('prompt_tokens', 0),
                    output_tokens=tokens.get('completion_tokens', 0),
                    cost_usd=cost,
                    response_time_ms=response_time * 1000,
                    success=success,
                    ip_hash=ip_hash,
                    user_agent=request.headers.get('User-Agent')
                )