"""
FastAPI Router Factory for NyanProxy

Creates FastAPI routers with proper service injection
Avoids circular import issues by creating routers on demand
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .handlers.openai_handler import OpenAIHandler
from .handlers.anthropic_handler import AnthropicHandler
from .handlers.health_handler import HealthHandler
from .handlers.system_handler import SystemHandler
from .handlers.google_handler import GoogleHandler
from .fastapi_security import require_auth
from .flask_compat import FlaskCompatibilityContext, convert_flask_response


# Pydantic models for request validation
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = None
    max_completion_tokens: int = None
    temperature: float = None
    stream: bool = False


def create_openai_router(services: Dict[str, Any]) -> APIRouter:
    """Create OpenAI router with service dependencies"""
    router = APIRouter(prefix="/v1", tags=["openai"])
    
    def get_openai_handler() -> OpenAIHandler:
        """Get OpenAI handler with injected services"""
        return OpenAIHandler(services)
    
    @router.post("/chat/completions", 
                 summary="OpenAI Chat Completions",
                 description="Create a chat completion with OpenAI models")
    async def chat_completions(
        request: Request,
        chat_request: ChatCompletionRequest,
        handler: OpenAIHandler = Depends(get_openai_handler),
        auth_data: Dict[str, Any] = Depends(require_auth)
    ):
        """Handle OpenAI chat completions with retry logic and validation"""
        try:
            # Use Flask compatibility context to ensure handler works
            with FlaskCompatibilityContext(request, chat_request.dict(), auth_data):
                result, status_code = handler.chat_completions()
            
            # Handle different response types
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
            
            # Check if it's a streaming response
            is_streaming = chat_request.stream
            
            if is_streaming and hasattr(result, 'iter_content'):
                # Handle streaming response
                def generate():
                    try:
                        for chunk in result.iter_content(chunk_size=1024):
                            if chunk:
                                yield chunk
                    except Exception as e:
                        yield f"data: {{'error': 'Stream error: {str(e)}'}}\n\n".encode()
                
                return StreamingResponse(
                    generate(),
                    media_type="text/plain" if not result.headers.get('content-type') else result.headers.get('content-type')
                )
            else:
                # Handle regular JSON response
                response_data = convert_flask_response(result, status_code)
                return response_data
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models",
                summary="List OpenAI Models", 
                description="List available whitelisted OpenAI models")
    async def list_models(
        request: Request,
        handler: OpenAIHandler = Depends(get_openai_handler)
    ):
        """Return whitelisted OpenAI models"""
        try:
            # Use Flask compatibility context
            with FlaskCompatibilityContext(request, {}, {}):
                result, status_code = handler.models()
            
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
            
            response_data = convert_flask_response(result, status_code)
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


# Pydantic models for Anthropic requests
class Message(BaseModel):
    role: str
    content: str


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int
    system: str = None
    temperature: float = None
    stream: bool = False


def create_anthropic_router(services: Dict[str, Any]) -> APIRouter:
    """Create Anthropic router with service dependencies"""
    router = APIRouter(prefix="/v1", tags=["anthropic"])
    
    def get_anthropic_handler() -> AnthropicHandler:
        """Get Anthropic handler with injected services"""
        return AnthropicHandler(services)
    
    @router.post("/messages", 
                 summary="Anthropic Messages",
                 description="Create a message completion with Anthropic models")
    async def messages(
        request: Request,
        message_request: AnthropicMessageRequest,
        handler: AnthropicHandler = Depends(get_anthropic_handler),
        auth_data: Dict[str, Any] = Depends(require_auth)
    ):
        """Handle Anthropic messages with retry logic and validation"""
        try:
            # Use Flask compatibility context to ensure handler works
            with FlaskCompatibilityContext(request, message_request.dict(), auth_data):
                result, status_code = handler.messages()
            
            # Handle different response types
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
            
            # Check if it's a streaming response
            is_streaming = message_request.stream
            
            if is_streaming and hasattr(result, 'iter_content'):
                # Handle streaming response
                def generate():
                    try:
                        for chunk in result.iter_content(chunk_size=1024):
                            if chunk:
                                yield chunk
                    except Exception as e:
                        yield f"data: {{'error': 'Stream error: {str(e)}'}}\\n\\n".encode()
                
                return StreamingResponse(
                    generate(),
                    media_type="text/plain" if not result.headers.get('content-type') else result.headers.get('content-type')
                )
            else:
                # Handle regular JSON response
                response_data = convert_flask_response(result, status_code)
                return response_data
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/models",
                summary="List Anthropic Models", 
                description="List available whitelisted Anthropic models")
    async def list_models(
        request: Request,
        handler: AnthropicHandler = Depends(get_anthropic_handler)
    ):
        """Return whitelisted Anthropic models"""
        try:
            # Use Flask compatibility context
            with FlaskCompatibilityContext(request, {}, {}):
                result, status_code = handler.models()
            
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
            
            response_data = convert_flask_response(result, status_code)
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


def create_health_router(services: Dict[str, Any]) -> APIRouter:
    """Create Health router with service dependencies"""
    router = APIRouter(prefix="/api", tags=["health"])
    
    def get_health_handler() -> HealthHandler:
        """Get Health handler with injected services"""
        return HealthHandler(services)
    
    @router.get("/health", 
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
    
    @router.get("/keys/status",
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
    
    @router.get("/keys/health",
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
    
    @router.get("/keys/debug",
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
    
    return router


def create_system_router(services: Dict[str, Any]) -> APIRouter:
    """Create System router with service dependencies"""
    router = APIRouter(prefix="/api", tags=["system"])
    
    def get_system_handler() -> SystemHandler:
        """Get System handler with injected services"""
        return SystemHandler(services)
    
    @router.get("/metrics", 
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
    
    @router.get("/system/health", 
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
    
    return router


# Pydantic models for Google requests
class GoogleGenerationConfig(BaseModel):
    maxOutputTokens: int = None
    temperature: float = None
    topP: float = None
    topK: int = None


class GoogleContentPart(BaseModel):
    text: str


class GoogleContent(BaseModel):
    role: str = "user"
    parts: list[GoogleContentPart]


class GoogleGeminiRequest(BaseModel):
    contents: list[GoogleContent]
    generationConfig: GoogleGenerationConfig = None
    stream: bool = False
    model: str = None


def create_google_router(services: Dict[str, Any]) -> APIRouter:
    """Create Google router with service dependencies"""
    router = APIRouter(prefix="/google", tags=["google"])
    
    def get_google_handler() -> GoogleHandler:
        """Get Google handler with injected services"""
        return GoogleHandler(services)
    
    @router.get("/v1/models",
                summary="List Google Models", 
                description="List available Google Gemini models")
    async def list_models(
        request: Request,
        handler: GoogleHandler = Depends(get_google_handler)
    ):
        """Return available Google models"""
        try:
            # Use Flask compatibility context
            with FlaskCompatibilityContext(request, {}, {}):
                result, status_code = handler.models()
            
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
            
            response_data = convert_flask_response(result, status_code)
            return response_data
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/v1/models/{model_path:path}",
                 summary="Google Gemini Chat",
                 description="Google Gemini API proxy endpoint for model interactions")
    async def gemini_proxy_with_model(
        request: Request,
        model_path: str,
        gemini_request: GoogleGeminiRequest,
        handler: GoogleHandler = Depends(get_google_handler),
        auth_data: Dict[str, Any] = Depends(require_auth)
    ):
        """Handle Google Gemini API proxy with model path"""
        try:
            # Use Flask compatibility context to ensure handler works
            with FlaskCompatibilityContext(request, gemini_request.dict(), auth_data):
                result, status_code = handler.gemini_proxy(model_path)
            
            # Handle different response types
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
            
            # Check if it's a streaming response
            is_streaming = gemini_request.stream
            
            if is_streaming and hasattr(result, 'iter_content'):
                # Handle streaming response
                def generate():
                    try:
                        for chunk in result.iter_content(chunk_size=1024):
                            if chunk:
                                yield chunk
                    except Exception as e:
                        yield f"data: {{'error': 'Stream error: {str(e)}'}}\\n\\n".encode()
                
                return StreamingResponse(
                    generate(),
                    media_type="text/plain" if not result.headers.get('content-type') else result.headers.get('content-type')
                )
            else:
                # Handle regular JSON response
                response_data = convert_flask_response(result, status_code)
                return response_data
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/v1/{model_path:path}",
                 summary="Google Gemini API (Alternative Path)",
                 description="Alternative Google Gemini API proxy endpoint")
    async def gemini_proxy_alternative(
        request: Request,
        model_path: str,
        gemini_request: GoogleGeminiRequest,
        handler: GoogleHandler = Depends(get_google_handler),
        auth_data: Dict[str, Any] = Depends(require_auth)
    ):
        """Handle Google Gemini API proxy with alternative path format"""
        return await gemini_proxy_with_model(request, model_path, gemini_request, handler, auth_data)
    
    return router


def create_all_routers(services: Dict[str, Any]) -> list:
    """Create all FastAPI routers with service dependencies"""
    routers = []
    
    # Create OpenAI router
    try:
        openai_router = create_openai_router(services)
        routers.append(("openai", openai_router))
        print("[ROUTER] OpenAI router created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI router: {e}")
    
    # Create Anthropic router
    try:
        anthropic_router = create_anthropic_router(services)
        routers.append(("anthropic", anthropic_router))
        print("[ROUTER] Anthropic router created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create Anthropic router: {e}")
    
    # Create Health router
    try:
        health_router = create_health_router(services)
        routers.append(("health", health_router))
        print("[ROUTER] Health router created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create Health router: {e}")
    
    # Create System router
    try:
        system_router = create_system_router(services)
        routers.append(("system", system_router))
        print("[ROUTER] System router created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create System router: {e}")
    
    # Create Google router
    try:
        google_router = create_google_router(services)
        routers.append(("google", google_router))
        print("[ROUTER] Google router created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create Google router: {e}")
    
    # Add other routers here as we convert them
    
    return routers