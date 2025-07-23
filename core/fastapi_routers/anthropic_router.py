"""
Anthropic FastAPI Router for NyanProxy

Converts Anthropic Handler to FastAPI router with full compatibility
Handles Anthropic messages, models, streaming, and all Anthropic API features
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..handlers.anthropic_handler import AnthropicHandler
from ..fastapi_security import require_auth
from ..flask_compat import FlaskCompatibilityContext, convert_flask_response


# Create FastAPI router
anthropic_router = APIRouter(prefix="/v1", tags=["anthropic"])


# Pydantic models for request/response validation
class Message(BaseModel):
    role: str
    content: str


class AnthropicMessageRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int
    system: Optional[str] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    # Add other Anthropic-specific parameters as needed


class AnthropicModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "anthropic"


class AnthropicModelsResponse(BaseModel):
    object: str = "list"
    data: List[AnthropicModelInfo]


async def get_anthropic_handler() -> AnthropicHandler:
    """Dependency to get Anthropic handler with services"""
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Get services from the global container
    from fastapi_app import services
    return AnthropicHandler(services)


@anthropic_router.post("/messages", 
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


@anthropic_router.get("/models",
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


# Legacy route support for OpenAI-style chat completions mapped to Anthropic messages
@anthropic_router.post("/chat/completions",
                      summary="Anthropic Chat Completions (Legacy)",
                      description="OpenAI-style chat completions using Anthropic models")
async def chat_completions_legacy(
    request: Request,
    message_request: AnthropicMessageRequest,
    handler: AnthropicHandler = Depends(get_anthropic_handler),
    auth_data: Dict[str, Any] = Depends(require_auth)
):
    """Legacy OpenAI-style chat completions endpoint using Anthropic"""
    # Convert to Anthropic messages format and call messages endpoint
    return await messages(request, message_request, handler, auth_data)