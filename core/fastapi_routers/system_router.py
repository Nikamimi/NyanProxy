"""
System FastAPI Router for NyanProxy

Converts System Handler to FastAPI router with full compatibility
Handles system metrics, monitoring, and diagnostic endpoints
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..handlers.system_handler import SystemHandler
from ..flask_compat import FlaskCompatibilityContext, convert_flask_response


# Create FastAPI router
system_router = APIRouter(prefix="/api", tags=["system"])


async def get_system_handler() -> SystemHandler:
    """Dependency to get system handler with services"""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Get services from the global container
    from fastapi_app import services
    return SystemHandler(services)


@system_router.get("/metrics", 
                   summary="System Metrics",
                   description="Get comprehensive proxy metrics")
async def get_metrics(
    request: Request,
    handler: SystemHandler = Depends(get_system_handler)
):
    """Get comprehensive proxy metrics"""
    try:
        # Use Flask compatibility context
        with FlaskCompatibilityContext(request, {}, {}):
            result, status_code = handler.get_metrics()
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
        
        response_data = convert_flask_response(result, status_code)
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@system_router.get("/system/health", 
                   summary="System Health",
                   description="Comprehensive system health endpoint with warnings and recommendations")
async def system_health(
    request: Request,
    handler: SystemHandler = Depends(get_system_handler)
):
    """Comprehensive system health endpoint with warnings and recommendations"""
    try:
        # Use Flask compatibility context
        with FlaskCompatibilityContext(request, {}, {}):
            result, status_code = handler.system_health()
        
        response_data = convert_flask_response(result, status_code)
        
        if status_code != 200:
            # For system health, even 500 status should return the health data
            raise HTTPException(status_code=status_code, detail=response_data)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))