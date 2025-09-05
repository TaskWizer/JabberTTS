"""Voice Cloning Studio API Routes.

This module provides REST API endpoints for the Voice Cloning Studio,
enabling web-based voice cloning and management capabilities.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import io
from pathlib import Path

from .studio_interface import VoiceCloningStudio

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/voice-cloning", tags=["Voice Cloning Studio"])

# Initialize templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global studio instance
_studio_instance: Optional[VoiceCloningStudio] = None


async def get_studio() -> VoiceCloningStudio:
    """Get or create the Voice Cloning Studio instance."""
    global _studio_instance
    if _studio_instance is None:
        _studio_instance = VoiceCloningStudio()
        await _studio_instance.initialize()
    return _studio_instance


# Request/Response Models

class VoiceUploadResponse(BaseModel):
    success: bool
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VoicePreviewRequest(BaseModel):
    voice_id: str
    preview_text: str = "Hello, this is a preview of my voice."
    modulation_params: Optional[Dict[str, float]] = None


class VoiceListRequest(BaseModel):
    category: Optional[str] = None
    language: Optional[str] = None
    gender: Optional[str] = None
    tags: Optional[List[str]] = None
    favorites_only: bool = False


class VoiceUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    is_favorite: Optional[bool] = None


class VoiceBlendRequest(BaseModel):
    voice_ids: List[str]
    weights: List[float]
    blend_name: str


class VoiceRecommendationRequest(BaseModel):
    text: str
    preferences: Optional[Dict[str, Any]] = None


# Web Interface

@router.get("/", response_class=HTMLResponse)
async def voice_cloning_studio():
    """Serve the Voice Cloning Studio web interface."""
    try:
        with open(Path(__file__).parent / "templates" / "studio.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Failed to serve studio interface: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load studio interface"
        )


# API Endpoints

@router.post("/upload", response_model=VoiceUploadResponse)
async def upload_voice_sample(
    file: UploadFile = File(...),
    voice_name: str = Form(...),
    description: str = Form(""),
    tags: str = Form(""),
    category: str = Form("custom")
):
    """Upload and process a voice sample for cloning.
    
    Args:
        file: Audio file to upload
        voice_name: Name for the voice
        description: Voice description
        tags: Comma-separated tags
        category: Voice category
        
    Returns:
        Voice upload response with analysis results
    """
    try:
        studio = await get_studio()
        
        # Read audio data
        audio_data = await file.read()
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Upload voice sample
        result = await studio.upload_voice_sample(
            audio_data=audio_data,
            voice_name=voice_name,
            description=description,
            tags=tag_list,
            category=category
        )
        
        return VoiceUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Voice upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice upload failed: {str(e)}"
        )


@router.post("/preview")
async def generate_voice_preview(request: VoicePreviewRequest):
    """Generate a preview of a cloned voice.
    
    Args:
        request: Voice preview request
        
    Returns:
        Audio preview response
    """
    try:
        studio = await get_studio()
        
        result = await studio.generate_voice_preview(
            voice_id=request.voice_id,
            preview_text=request.preview_text,
            modulation_params=request.modulation_params
        )
        
        if result["success"]:
            # Return audio as streaming response
            audio_data = result["audio_data"]
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"attachment; filename=voice_preview_{request.voice_id}.mp3"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice preview generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice preview generation failed: {str(e)}"
        )


@router.post("/list")
async def list_voices(request: VoiceListRequest):
    """List available voices with optional filtering.
    
    Args:
        request: Voice list request with filters
        
    Returns:
        List of voices matching the filters
    """
    try:
        studio = await get_studio()
        
        filters = {
            "category": request.category,
            "language": request.language,
            "gender": request.gender,
            "tags": request.tags,
            "favorites_only": request.favorites_only
        }
        
        result = await studio.list_voices(filters)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Voice listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice listing failed: {str(e)}"
        )


@router.get("/search")
async def search_voices(query: str):
    """Search voices by name, description, or tags.
    
    Args:
        query: Search query
        
    Returns:
        Search results
    """
    try:
        studio = await get_studio()
        
        result = await studio.search_voices(query)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Voice search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice search failed: {str(e)}"
        )


@router.get("/voice/{voice_id}")
async def get_voice_details(voice_id: str):
    """Get detailed information about a voice.
    
    Args:
        voice_id: Voice ID
        
    Returns:
        Detailed voice information
    """
    try:
        studio = await get_studio()
        
        result = await studio.get_voice_details(voice_id)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Get voice details failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Get voice details failed: {str(e)}"
        )


@router.put("/voice/{voice_id}")
async def update_voice_metadata(voice_id: str, request: VoiceUpdateRequest):
    """Update voice metadata.
    
    Args:
        voice_id: Voice ID
        request: Voice update request
        
    Returns:
        Update result
    """
    try:
        studio = await get_studio()
        
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.tags is not None:
            updates["tags"] = request.tags
        if request.category is not None:
            updates["category"] = request.category
        if request.is_favorite is not None:
            updates["is_favorite"] = request.is_favorite
        
        result = await studio.update_voice_metadata(voice_id, updates)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Voice metadata update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice metadata update failed: {str(e)}"
        )


@router.delete("/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice from the library.
    
    Args:
        voice_id: Voice ID
        
    Returns:
        Deletion result
    """
    try:
        studio = await get_studio()
        
        result = await studio.delete_voice(voice_id)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Voice deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice deletion failed: {str(e)}"
        )


@router.get("/voice/{voice_id}/similar")
async def find_similar_voices(voice_id: str, max_results: int = 5):
    """Find voices similar to the specified voice.
    
    Args:
        voice_id: Reference voice ID
        max_results: Maximum number of results
        
    Returns:
        Similar voices
    """
    try:
        studio = await get_studio()
        
        result = await studio.find_similar_voices(voice_id, max_results)
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Similar voices search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar voices search failed: {str(e)}"
        )


@router.post("/recommendations")
async def get_voice_recommendations(request: VoiceRecommendationRequest):
    """Get voice recommendations for given text and preferences.
    
    Args:
        request: Voice recommendation request
        
    Returns:
        Voice recommendations
    """
    try:
        studio = await get_studio()
        
        result = await studio.get_voice_recommendations(
            request.text,
            request.preferences
        )
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Voice recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice recommendations failed: {str(e)}"
        )


@router.post("/blend")
async def create_voice_blend(request: VoiceBlendRequest):
    """Create a blended voice from multiple source voices.
    
    Args:
        request: Voice blend request
        
    Returns:
        Blend result
    """
    try:
        studio = await get_studio()
        
        result = await studio.create_voice_blend(
            request.voice_ids,
            request.weights,
            request.blend_name
        )
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Voice blending failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice blending failed: {str(e)}"
        )


@router.get("/statistics")
async def get_library_statistics():
    """Get voice library statistics.
    
    Returns:
        Library statistics
    """
    try:
        studio = await get_studio()
        
        result = await studio.get_library_statistics()
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Library statistics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Library statistics failed: {str(e)}"
        )


@router.post("/synthesize")
async def synthesize_with_cloned_voice(
    voice_id: str = Form(...),
    text: str = Form(...),
    output_format: str = Form("mp3"),
    speed: float = Form(1.0),
    modulation_params: Optional[str] = Form(None)
):
    """Synthesize speech using a cloned voice.
    
    Args:
        voice_id: ID of the cloned voice
        text: Text to synthesize
        output_format: Output audio format
        speed: Speech speed
        modulation_params: JSON string of modulation parameters
        
    Returns:
        Synthesized audio
    """
    try:
        studio = await get_studio()
        
        # Parse modulation parameters if provided
        mod_params = None
        if modulation_params:
            import json
            mod_params = json.loads(modulation_params)
        
        # Generate preview first to get audio data
        result = await studio.generate_voice_preview(
            voice_id=voice_id,
            preview_text=text,
            modulation_params=mod_params
        )
        
        if result["success"]:
            audio_data = result["audio_data"]
            
            # Determine content type
            content_type = "audio/mpeg" if output_format == "mp3" else f"audio/{output_format}"
            
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=synthesized_{voice_id}.{output_format}"
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice synthesis failed: {str(e)}"
        )
