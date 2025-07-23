"""
Google FastAPI Router for NyanProxy

Converts Google Handler to FastAPI router with full compatibility
Handles Google Gemini API proxying, models, and all Google API features
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..handlers.google_handler import GoogleHandler
from ..fastapi_security import require_auth
from ..flask_compat import FlaskCompatibilityContext, convert_flask_response


# Create FastAPI router
google_router = APIRouter(prefix="/google", tags=["google"])


# Pydantic models for request/response validation
class GoogleGenerationConfig(BaseModel):
    maxOutputTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    topK: Optional[int] = None


class GoogleContentPart(BaseModel):
    text: str


class GoogleContent(BaseModel):
    role: Optional[str] = "user"
    parts: list[GoogleContentPart]


class GoogleGeminiRequest(BaseModel):
    contents: list[GoogleContent]
    generationConfig: Optional[GoogleGenerationConfig] = None
    stream: Optional[bool] = False
    model: Optional[str] = None  # Sometimes included in request body


class GoogleModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class GoogleModelsResponse(BaseModel):
    object: str = "list"
    data: list[GoogleModelInfo]


async def get_google_handler() -> GoogleHandler:
    """Dependency to get Google handler with services"""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Get services from the global container
    from fastapi_app import services
    return GoogleHandler(services)


@google_router.get("/v1/models",
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


@google_router.post("/v1/models/{model_path:path}",
                    summary="Google Gemini Chat",
                    description="Google Gemini API proxy endpoint for model interactions")
async def gemini_proxy_with_model(
    request: Request,
    model_path: str = Path(..., description="Google model path"),
    gemini_request: GoogleGeminiRequest = None,
    handler: GoogleHandler = Depends(get_google_handler),
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """Handle Google Gemini API proxy with model path"""
    try:
        # Convert request to dict, handling case where body might be empty
        request_data = gemini_request.dict() if gemini_request else {}
        
        # Use Flask compatibility context to ensure handler works
        with FlaskCompatibilityContext(request, request_data, auth_data):
            result, status_code = handler.gemini_proxy(model_path)
        
        # Handle different response types
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=convert_flask_response(result, status_code))
        
        # Check if it's a streaming response
        is_streaming = request_data.get('stream', False) if request_data else False
        
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


@google_router.post("/v1/{model_path:path}",
                    summary="Google Gemini API (Alternative Path)",
                    description="Alternative Google Gemini API proxy endpoint")
async def gemini_proxy_alternative(
    request: Request,
    model_path: str = Path(..., description="Google API path"),
    gemini_request: GoogleGeminiRequest = None,
    handler: GoogleHandler = Depends(get_google_handler),
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """Handle Google Gemini API proxy with alternative path format"""
    return await gemini_proxy_with_model(request, model_path, gemini_request, handler, auth_data)


# Additional compatibility endpoint for direct model access
@google_router.post("/models/{model_name}:generateContent",
                    summary="Google Generate Content",
                    description="Direct Google model content generation endpoint")
async def generate_content(
    request: Request,
    model_name: str = Path(..., description="Google model name"),
    gemini_request: GoogleGeminiRequest = None,
    handler: GoogleHandler = Depends(get_google_handler),
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """Direct Google model content generation"""
    # Construct the full model path for the handler
    model_path = f"models/{model_name}:generateContent"
    return await gemini_proxy_with_model(request, model_path, gemini_request, handler, auth_data)