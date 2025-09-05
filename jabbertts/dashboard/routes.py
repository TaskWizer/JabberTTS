"""JabberTTS Dashboard Routes.

This module provides web-based dashboard routes for testing and demonstrating
JabberTTS capabilities.
"""

import logging
import base64
from typing import Dict, Any
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from jabbertts.config import get_settings
from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.metrics import get_metrics_collector
from jabbertts.validation import (
    get_whisper_validator,
    get_validation_metrics,
    get_self_debugger,
    get_validation_test_suite
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page."""
    settings = get_settings()
    
    # Get system information
    audio_processor = get_audio_processor()
    processor_info = audio_processor.get_processor_info()
    
    context = {
        "request": request,
        "title": "JabberTTS Dashboard",
        "version": "0.1.0",
        "settings": {
            "audio_quality": settings.audio_quality,
            "enable_audio_enhancement": settings.enable_audio_enhancement,
            "sample_rate": settings.sample_rate,
            "default_voice": settings.default_voice,
            "default_format": settings.default_format,
        },
        "processor_info": processor_info,
        "voices": [
            {"id": "alloy", "name": "Alloy", "description": "Balanced, neutral voice"},
            {"id": "echo", "name": "Echo", "description": "Clear, articulate voice"},
            {"id": "fable", "name": "Fable", "description": "Warm, storytelling voice"},
            {"id": "onyx", "name": "Onyx", "description": "Deep, authoritative voice"},
            {"id": "nova", "name": "Nova", "description": "Bright, energetic voice"},
            {"id": "shimmer", "name": "Shimmer", "description": "Gentle, soothing voice"},
        ],
        "formats": [
            {"id": "mp3", "name": "MP3", "description": "Good compression, widely supported"},
            {"id": "wav", "name": "WAV", "description": "Uncompressed, high quality"},
            {"id": "flac", "name": "FLAC", "description": "Lossless compression"},
            {"id": "opus", "name": "Opus", "description": "Efficient compression for streaming"},
            {"id": "aac", "name": "AAC", "description": "Good compression, mobile-friendly"},
            {"id": "pcm", "name": "PCM", "description": "Raw audio data"},
        ],
        "quality_presets": [
            {"id": "low", "name": "Low", "description": "16kHz, basic quality"},
            {"id": "standard", "name": "Standard", "description": "24kHz, good quality"},
            {"id": "high", "name": "High", "description": "44.1kHz, high quality"},
            {"id": "ultra", "name": "Ultra", "description": "48kHz, maximum quality"},
        ]
    }
    
    return templates.TemplateResponse("dashboard.html", context)


@router.post("/generate")
async def generate_speech_api(
    text: str = Form(...),
    voice: str = Form("alloy"),
    format: str = Form("mp3"),
    speed: float = Form(1.0),
    quality: str = Form("standard")
):
    """Generate speech via dashboard API."""
    try:
        logger.info(f"Dashboard TTS request: {len(text)} chars, voice={voice}, format={format}")
        
        # Get inference engine and audio processor
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        
        # Generate speech
        result = await inference_engine.generate_speech(
            text=text,
            voice=voice,
            speed=speed,
            response_format=format
        )
        
        # Process audio
        audio_data, audio_metadata = await audio_processor.process_audio(
            audio_array=result["audio_data"],
            sample_rate=result["sample_rate"],
            output_format=format,
            speed=speed,
            original_sample_rate=result["sample_rate"]
        )

        # Encode audio data as base64 for JSON response
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Determine content type
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        content_type = content_type_map.get(format, "audio/mpeg")

        # Return audio data and metadata
        return JSONResponse({
            "success": True,
            "audio_data": audio_base64,
            "content_type": content_type,
            "audio_size": len(audio_data),
            "duration": audio_metadata["processed_duration"],  # Use actual processed duration
            "raw_duration": result["duration"],  # Original TTS duration for RTF calculation
            "sample_rate": audio_metadata["final_sample_rate"],  # Final sample rate after processing
            "original_sample_rate": result["sample_rate"],  # Original TTS sample rate
            "format": format,
            "voice": voice,
            "speed": speed,
            "text_length": len(text),
            "rtf": result.get("rtf", 0),
            "inference_time": result.get("inference_time", 0),
            "enhancement_applied": audio_metadata["enhancement_applied"]
        })
        
    except Exception as e:
        logger.error(f"Dashboard TTS generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech generation failed: {str(e)}"
        )


@router.post("/download")
async def download_audio(
    text: str = Form(...),
    voice: str = Form("alloy"),
    format: str = Form("mp3"),
    speed: float = Form(1.0),
    quality: str = Form("standard")
):
    """Download generated audio file directly."""
    try:
        logger.info(f"Dashboard audio download: {len(text)} chars, voice={voice}, format={format}")

        # Get inference engine and audio processor
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()

        # Generate speech
        result = await inference_engine.generate_speech(
            text=text,
            voice=voice,
            speed=speed,
            response_format=format
        )

        # Process audio
        audio_data = await audio_processor.process_audio(
            audio_array=result["audio_data"],
            sample_rate=result["sample_rate"],
            output_format=format,
            speed=speed
        )

        # Determine content type
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        content_type = content_type_map.get(format, "audio/mpeg")

        # Return audio file
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=jabbertts_speech.{format}",
                "Content-Length": str(len(audio_data))
            }
        )

    except Exception as e:
        logger.error(f"Dashboard audio download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio download failed: {str(e)}"
        )


@router.get("/api/status")
async def get_system_status():
    """Get system status for dashboard."""
    try:
        settings = get_settings()
        audio_processor = get_audio_processor()
        
        return JSONResponse({
            "status": "healthy",
            "version": "0.1.0",
            "audio_processor": audio_processor.get_processor_info(),
            "settings": {
                "audio_quality": settings.audio_quality,
                "enable_audio_enhancement": settings.enable_audio_enhancement,
                "sample_rate": settings.sample_rate,
                "host": settings.host,
                "port": settings.port,
            }
        })
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status check failed: {str(e)}"
        )


@router.get("/api/system")
async def get_system_metrics():
    """Get detailed system metrics for dashboard."""
    try:
        metrics_collector = get_metrics_collector()
        system_status = metrics_collector.get_system_status()

        return JSONResponse(system_status)

    except Exception as e:
        logger.error(f"System metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System metrics failed: {str(e)}"
        )


@router.get("/api/validation/summary")
async def get_validation_summary():
    """Get validation system summary."""
    try:
        validation_metrics = get_validation_metrics()
        summary = validation_metrics.get_validation_summary(window_minutes=60)

        return JSONResponse(summary)

    except Exception as e:
        logger.error(f"Validation summary failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation summary failed: {str(e)}"
        )


@router.get("/api/validation/health")
async def get_validation_health():
    """Get validation system health assessment."""
    try:
        debugger = get_self_debugger()
        health = debugger.get_issue_summary(window_minutes=60)

        return JSONResponse(health)

    except Exception as e:
        logger.error(f"Validation health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation health check failed: {str(e)}"
        )


@router.get("/api/validation/diagnosis")
async def get_validation_diagnosis():
    """Get full validation system diagnosis."""
    try:
        debugger = get_self_debugger()
        diagnosis = debugger.run_full_diagnosis(window_minutes=60)

        return JSONResponse(diagnosis)

    except Exception as e:
        logger.error(f"Validation diagnosis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation diagnosis failed: {str(e)}"
        )


@router.post("/api/validation/test")
async def run_validation_test(
    text: str = Form(...),
    voice: str = Form("alloy"),
    format: str = Form("mp3"),
    speed: float = Form(1.0)
):
    """Run a single validation test."""
    try:
        logger.info(f"Running validation test: {len(text)} chars, voice={voice}")

        # Generate TTS audio
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()

        # Generate speech
        tts_result = await inference_engine.generate_speech(
            text=text,
            voice=voice,
            speed=speed,
            response_format=format
        )

        # Process audio
        audio_data = await audio_processor.process_audio(
            audio_array=tts_result["audio_data"],
            sample_rate=tts_result["sample_rate"],
            output_format=format,
            speed=speed
        )

        # Validate with Whisper
        validator = get_whisper_validator("tiny")  # Use tiny model for faster testing
        validation_result = validator.validate_tts_output(text, audio_data, tts_result["sample_rate"])

        if not validation_result["success"]:
            return JSONResponse({
                "success": False,
                "error": validation_result.get("error", "Validation failed")
            })

        # Record validation metrics
        validation_metrics = get_validation_metrics()
        validation_metrics.record_validation(
            test_category="manual",
            voice=voice,
            format=format,
            speed=speed,
            success=True,
            accuracy_score=validation_result["accuracy_metrics"]["overall_accuracy"],
            quality_score=0.8,  # Placeholder - would come from quality assessment
            rtf=tts_result.get("rtf", 0),
            inference_time=tts_result.get("inference_time", 0),
            validation_time=validation_result.get("transcription_info", {}).get("transcription_time", 0)
        )

        return JSONResponse({
            "success": True,
            "validation_result": validation_result,
            "tts_metrics": {
                "rtf": tts_result.get("rtf", 0),
                "inference_time": tts_result.get("inference_time", 0),
                "audio_duration": tts_result.get("audio_duration", 0)
            }
        })

    except Exception as e:
        logger.error(f"Validation test failed: {e}")

        # Record failed validation
        try:
            validation_metrics = get_validation_metrics()
            validation_metrics.record_validation(
                test_category="manual",
                voice=voice,
                format=format,
                speed=speed,
                success=False,
                accuracy_score=0.0,
                quality_score=0.0,
                error_message=str(e)
            )
        except:
            pass  # Don't fail if metrics recording fails

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation test failed: {str(e)}"
        )


@router.post("/api/validation/quick-test")
async def run_quick_validation():
    """Run quick validation test suite."""
    try:
        logger.info("Running quick validation test suite")

        test_suite = get_validation_test_suite("tiny")  # Use tiny model for speed
        results = await test_suite.run_quick_validation(sample_count=3)

        return JSONResponse(results)

    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick validation failed: {str(e)}"
        )


@router.get("/api/voices")
async def get_available_voices():
    """Get available voices for dashboard."""
    voices = [
        {
            "id": "alloy",
            "name": "Alloy",
            "description": "A balanced, neutral voice",
            "type": "built-in",
            "preview_text": "Hello, I'm Alloy. I have a balanced and neutral tone."
        },
        {
            "id": "echo",
            "name": "Echo",
            "description": "A clear, articulate voice",
            "type": "built-in",
            "preview_text": "Hello, I'm Echo. I speak with clarity and articulation."
        },
        {
            "id": "fable",
            "name": "Fable",
            "description": "A warm, storytelling voice",
            "type": "built-in",
            "preview_text": "Hello, I'm Fable. I love telling stories with warmth."
        },
        {
            "id": "onyx",
            "name": "Onyx",
            "description": "A deep, authoritative voice",
            "type": "built-in",
            "preview_text": "Hello, I'm Onyx. I speak with depth and authority."
        },
        {
            "id": "nova",
            "name": "Nova",
            "description": "A bright, energetic voice",
            "type": "built-in",
            "preview_text": "Hello, I'm Nova! I'm bright and full of energy!"
        },
        {
            "id": "shimmer",
            "name": "Shimmer",
            "description": "A gentle, soothing voice",
            "type": "built-in",
            "preview_text": "Hello, I'm Shimmer. I speak with gentle, soothing tones."
        }
    ]
    
    return JSONResponse({"voices": voices})


@router.get("/api/performance")
async def get_performance_metrics():
    """Get real-time performance metrics for dashboard."""
    try:
        metrics_collector = get_metrics_collector()
        metrics = metrics_collector.get_performance_metrics(window_minutes=10)

        return JSONResponse(metrics)

    except Exception as e:
        logger.error(f"Performance metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics failed: {str(e)}"
        )


@router.get("/api/metrics/validate")
async def validate_metrics_consistency():
    """Validate metrics consistency across the system."""
    try:
        metrics_collector = get_metrics_collector()
        validation_results = metrics_collector.validate_metrics_consistency()

        return JSONResponse(validation_results)

    except Exception as e:
        logger.error(f"Metrics validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics validation failed: {str(e)}"
        )
