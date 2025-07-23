"""
OpenAI FastAPI Router for NyanProxy

Converts OpenAI Handler to FastAPI router with full compatibility
Handles chat completions, models, streaming, and all OpenAI API features
"""
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..handlers.openai_handler import OpenAIHandler
from ..fastapi_security import require_auth
from ..flask_compat import FlaskCompatibilityContext, convert_flask_response


# Create FastAPI router
openai_router = APIRouter(prefix="/v1", tags=["openai"])


# Pydantic models for request/response validation
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    # Add other OpenAI parameters as needed


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


async def get_openai_handler() -> OpenAIHandler:
    """Dependency to get OpenAI handler with services"""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Get services from the global container
    from fastapi_app import services
    return OpenAIHandler(services)


# Flask response conversion is now handled in flask_compat module


@openai_router.post("/chat/completions", 
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


@openai_router.get("/models",
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


# Legacy route support (as mentioned in CLAUDE.md)
@openai_router.post("/completions",
                   summary="OpenAI Completions (Legacy)",
                   description="Legacy completions endpoint - redirects to chat completions")
async def completions(
    request: Request,
    chat_request: ChatCompletionRequest,
    handler: OpenAIHandler = Depends(get_openai_handler),
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """Legacy completions endpoint - redirects to chat completions"""
    # Convert completions format to chat completions format if needed
    return await chat_completions(request, chat_request, handler, auth_data)