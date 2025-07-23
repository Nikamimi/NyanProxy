"""
Route Handlers Package for NyanProxy

Contains extracted route handlers organized by functionality
Perfect for FastAPI migration - each handler becomes a router
"""
from .base_handler import BaseHandler
from .openai_handler import OpenAIHandler
from .anthropic_handler import AnthropicHandler
from .health_handler import HealthHandler
from .system_handler import SystemHandler

__all__ = ['BaseHandler', 'OpenAIHandler', 'AnthropicHandler', 'HealthHandler', 'SystemHandler']