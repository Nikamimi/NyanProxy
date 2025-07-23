"""
FastAPI Routers Package for NyanProxy

Converts extracted Flask handlers to FastAPI routers
Clean separation between business logic and web framework
"""
from .health_router import health_router
from .system_router import system_router

__all__ = ['health_router', 'system_router']