"""JabberTTS API Routes.

This module defines all API endpoints for the JabberTTS service,
including the main speech generation endpoint and voice management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from jabbertts.api.models import (
    TTSRequest,
    TTSResponse,
    VoiceListResponse,
    ErrorResponse
)
from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor

logger = logging.getLogger(__name__)

# Create the main API router
router = APIRouter()


@router.post(
    "/audio/speech",
    response_class=Response,
    responses={
        200: {"description": "Audio file generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate speech from text",
    description="Convert input text to speech audio using the specified voice and model."
)
async def create_speech(request: TTSRequest) -> Response:
    """Generate speech audio from input text.
    
    This endpoint is fully compatible with OpenAI's TTS API.
    
    Args:
        request: TTS request containing text, voice, and other parameters
        
    Returns:
        Audio response in the specified format
        
    Raises:
        HTTPException: If request is invalid or generation fails
    """
    try:
        logger.info(f"Generating speech for text length: {len(request.input)}")

        # Get inference engine and audio processor
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()

        # Generate speech using the inference engine
        result = await inference_engine.generate_speech(
            text=request.input,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format
        )

        # Process audio to the requested format
        audio_data = await audio_processor.process_audio(
            audio_array=result["audio_data"],
            sample_rate=result["sample_rate"],
            output_format=request.response_format,
            speed=request.speed
        )
        
        # Determine content type based on format
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        content_type = content_type_map.get(request.response_format, "audio/mpeg")
        
        logger.info(f"Generated audio for voice '{request.voice}' in format '{request.response_format}'")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during speech generation"
        )


@router.get(
    "/voices",
    response_model=VoiceListResponse,
    summary="List available voices",
    description="Get a list of all available voices including built-in and custom voices."
)
async def list_voices() -> VoiceListResponse:
    """List all available voices.
    
    Returns:
        List of available voices with metadata
    """
    try:
        # TODO: Implement actual voice listing
        # This is a placeholder that will be replaced with real implementation
        
        # Default OpenAI-compatible voices
        default_voices = [
            {
                "id": "alloy",
                "name": "Alloy",
                "description": "A balanced, neutral voice",
                "type": "built-in"
            },
            {
                "id": "echo",
                "name": "Echo",
                "description": "A clear, articulate voice",
                "type": "built-in"
            },
            {
                "id": "fable",
                "name": "Fable",
                "description": "A warm, storytelling voice",
                "type": "built-in"
            },
            {
                "id": "onyx",
                "name": "Onyx",
                "description": "A deep, authoritative voice",
                "type": "built-in"
            },
            {
                "id": "nova",
                "name": "Nova",
                "description": "A bright, energetic voice",
                "type": "built-in"
            },
            {
                "id": "shimmer",
                "name": "Shimmer",
                "description": "A gentle, soothing voice",
                "type": "built-in"
            }
        ]
        
        logger.info(f"Listed {len(default_voices)} available voices")
        
        return VoiceListResponse(voices=default_voices)
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing voices"
        )


@router.post(
    "/voices",
    status_code=status.HTTP_201_CREATED,
    summary="Upload custom voice",
    description="Upload an audio sample to create a custom voice for cloning."
)
async def create_voice():
    """Create a custom voice from uploaded audio sample.
    
    This endpoint will be implemented in the voice cloning phase.
    """
    # TODO: Implement voice cloning upload
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Voice cloning feature is not yet implemented"
    )


@router.delete(
    "/voices/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete custom voice",
    description="Delete a custom voice by ID."
)
async def delete_voice(voice_id: str):
    """Delete a custom voice.
    
    Args:
        voice_id: ID of the voice to delete
    """
    # TODO: Implement voice deletion
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Voice management feature is not yet implemented"
    )


@router.get(
    "/models",
    summary="List available models",
    description="Get a list of available TTS models."
)
async def list_models():
    """List available TTS models.
    
    Returns:
        List of available models
    """
    try:
        models = [
            {
                "id": "openaudio-s1-mini",
                "name": "OpenAudio S1 Mini",
                "description": "Optimized OpenAudio S1 model for CPU inference",
                "type": "neural",
                "languages": ["en", "multilingual"],
                "capabilities": ["tts", "voice_cloning"]
            }
        ]
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while listing models"
        )
