"""
Health FastAPI Router for NyanProxy

Converts Health Handler to FastAPI router with full compatibility
Handles health checks, key status monitoring, and debugging endpoints
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..handlers.health_handler import HealthHandler
from ..flask_compat import FlaskCompatibilityContext, convert_flask_response


# Create FastAPI router
health_router = APIRouter(prefix="/api", tags=["health"])


async def get_health_handler() -> HealthHandler:
    """Dependency to get health handler with services"""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Get services from the global container
    from fastapi_app import services
    return HealthHandler(services)


@health_router.get("/health", 
                   summary="Basic Health Check",
                   description="Basic health check endpoint for service status")
async def basic_health(
    request: Request,
    handler: HealthHandler = Depends(get_health_handler)
):
    """Basic health check endpoint"""
    try:
        # Use Flask compatibility context
        with FlaskCompatibilityContext(request, {}, {}):
            result, status_code = handler.basic_health()
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
        
        response_data = convert_flask_response(result, status_code)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@health_router.get("/keys/status",
                   summary="API Key Status",
                   description="Check which API keys are configured")
async def key_status(
    request: Request,
    handler: HealthHandler = Depends(get_health_handler)
):
    """Check which API keys are configured"""
    try:
        # Use Flask compatibility context
        with FlaskCompatibilityContext(request, {}, {}):
            result, status_code = handler.key_status()
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
        
        response_data = convert_flask_response(result, status_code)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@health_router.get("/keys/health",
                   summary="API Key Health",
                   description="Get detailed health status for all keys")
async def key_health(
    request: Request,
    handler: HealthHandler = Depends(get_health_handler)
):
    """Get detailed health status for all keys"""
    try:
        # Use Flask compatibility context
        with FlaskCompatibilityContext(request, {}, {}):
            result, status_code = handler.key_health()
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
        
        response_data = convert_flask_response(result, status_code)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@health_router.get("/keys/debug",
                   summary="Debug Key Loading",
                   description="Debug endpoint to check key loading")
async def debug_keys(
    request: Request,
    handler: HealthHandler = Depends(get_health_handler)
):
    """Debug endpoint to check key loading"""
    try:
        # Use Flask compatibility context
        with FlaskCompatibilityContext(request, {}, {}):
            result, status_code = handler.debug_keys()
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
        
        response_data = convert_flask_response(result, status_code)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))