"""
FastAPI Application for NyanProxy

Modern async replacement for Flask application
Uses extracted services and handlers from refactoring phase
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our extracted services
from core.services.thread_manager import ThreadManager
from core.services.connection_pool import ConnectionPoolManager
from core.services.metrics_tracker import MetricsTracker
from core.services.legacy_api_key_manager import LegacyAPIKeyManager

# Import business services
from src.services.model_families import ModelFamilyManager
from src.services.user_store import UserStore
from src.services.firebase_logger import FirebaseStructuredLogger
from src.services.conversation_logger import ConversationLogger

# Global services container
services: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown"""
    # Startup
    print("[STARTUP] Initializing NyanProxy FastAPI services...")
    
    # Initialize core services
    thread_manager = ThreadManager()
    connection_pool = ConnectionPoolManager()
    metrics = MetricsTracker(thread_manager)
    key_manager = LegacyAPIKeyManager()
    
    # Initialize business services  
    model_manager = ModelFamilyManager()
    user_store = UserStore()
    firebase_logger = FirebaseStructuredLogger()
    conversation_logger = ConversationLogger()
    
    # Store in global services container
    global services
    services = {
        'thread_manager': thread_manager,
        'connection_pool': connection_pool,
        'metrics': metrics,
        'key_manager': key_manager,
        'model_manager': model_manager,
        'user_store': user_store,
        'event_logger': conversation_logger,
        'structured_logger': firebase_logger
    }
    
    print("[STARTUP] All services initialized successfully")
    print(f"[STARTUP] Thread manager: {len(thread_manager.active_threads)} active threads")
    print(f"[STARTUP] Key manager: {len(key_manager.api_keys)} service pools loaded")
    print(f"[STARTUP] Model manager: Initialized with dynamic model system")
    
    # Now that services are ready, create and mount routers
    print("[STARTUP] Creating and mounting FastAPI routers...")
    try:
        from core.router_factory import create_all_routers
        routers = create_all_routers(services)
        
        for router_name, router in routers:
            app.include_router(router)
            print(f"[STARTUP] {router_name} router mounted successfully")
        
        print(f"[STARTUP] Total routers mounted: {len(routers)}")
    except Exception as e:
        print(f"[ERROR] Failed to create/mount routers: {e}")
        import traceback
        traceback.print_exc()
    
    print("[STARTUP] FastAPI application ready for requests")
    
    yield
    
    # Shutdown
    print("[SHUTDOWN] Cleaning up NyanProxy FastAPI services...")
    
    # Cleanup thread manager
    if 'thread_manager' in services:
        services['thread_manager'].cleanup_all_threads()
        print("[SHUTDOWN] Thread manager cleaned up")
    
    # Cleanup connection pools
    if 'connection_pool' in services:
        # Connection pools will be cleaned up automatically by requests
        print("[SHUTDOWN] Connection pools closed")
    
    # Force garbage collection
    if 'metrics' in services:
        collected = services['metrics'].force_garbage_collection()
        print(f"[SHUTDOWN] Garbage collection: {collected} objects collected")
    
    print("[SHUTDOWN] FastAPI shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="NyanProxy AI API Gateway",
    description="Unified AI proxy service with intelligent key management and retry logic",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Import security module
from core.fastapi_security import check_rate_limit

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests"""
    try:
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            response = await call_next(request)
            return response
        
        # Check rate limit
        await check_rate_limit(request)
        response = await call_next(request)
        return response
    except Exception as e:
        # If rate limit check fails, let it through but log
        print(f"Rate limit middleware error: {e}")
        response = await call_next(request)
        return response


# Dependency injection functions
async def get_thread_manager() -> ThreadManager:
    """Get thread manager service"""
    return services['thread_manager']


async def get_connection_pool() -> ConnectionPoolManager:
    """Get connection pool service"""
    return services['connection_pool']


async def get_metrics() -> MetricsTracker:
    """Get metrics tracker service"""
    return services['metrics']


async def get_key_manager() -> LegacyAPIKeyManager:
    """Get API key manager service"""
    return services['key_manager']


async def get_model_manager() -> ModelFamilyManager:
    """Get model family manager service"""
    return services['model_manager']


async def get_user_store() -> UserStore:
    """Get user store service"""
    return services['user_store']


async def get_event_logger() -> ConversationLogger:
    """Get event logger service"""
    return services['event_logger']


async def get_structured_logger() -> FirebaseStructuredLogger:
    """Get structured logger service"""
    return services['structured_logger']


async def get_services() -> Dict[str, Any]:
    """Get all services for handler dependency injection"""
    return services


# Basic health check endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "NyanProxy AI Gateway",
        "version": "2.0.0",
        "status": "running",
        "framework": "FastAPI",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(metrics: MetricsTracker = Depends(get_metrics)):
    """Basic health check"""
    uptime = metrics.get_uptime()
    return {
        "status": "healthy",
        "service": "AI Proxy",
        "uptime_seconds": uptime,
        "framework": "FastAPI"
    }


# FastAPI routers will be mounted during startup after services are initialized


# Test endpoint to verify service injection
@app.get("/api/services/status")
async def services_status(
    metrics: MetricsTracker = Depends(get_metrics),
    key_manager: LegacyAPIKeyManager = Depends(get_key_manager),
    model_manager: ModelFamilyManager = Depends(get_model_manager)
):
    """Get status of all services"""
    return {
        "metrics": {
            "uptime_seconds": metrics.get_uptime(),
            "total_requests": metrics.total_requests,
            "active_ips": len(metrics.active_ips)
        },
        "key_manager": {
            "total_services": len(key_manager.api_keys),
            "healthy_keys": sum(len(keys) for keys in key_manager.api_keys.values())
        },
        "model_manager": {
            "status": "initialized",
            "dynamic_system": True
        },
        "services_loaded": list(services.keys()),
        "mounted_routers": ["dynamically_mounted"]
    }


if __name__ == "__main__":
    # Development server
    print("Starting NyanProxy FastAPI development server...")
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )