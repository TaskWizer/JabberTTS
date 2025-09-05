"""JabberTTS Dashboard Routes.

This module provides web-based dashboard routes for testing and demonstrating
JabberTTS capabilities.
"""

import logging
import base64
import io
import json
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException, Form, status, File, UploadFile
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


@router.post("/debug/transcribe")
async def debug_transcribe_audio(
    audio_file: UploadFile = File(...),
    original_text: Optional[str] = Form(None)
):
    """Debug endpoint to transcribe uploaded audio and compare with original text.

    This endpoint provides detailed transcription analysis including:
    - Whisper STT transcription of uploaded audio
    - Word Error Rate (WER) and Character Error Rate (CER) metrics
    - Confidence scores per word/phrase
    - Side-by-side comparison with original text
    - Audio quality analysis
    """
    try:
        logger.info(f"Debug transcription request: {audio_file.filename}, original_text: {bool(original_text)}")

        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an audio file"
            )

        # Read audio data
        audio_data = await audio_file.read()

        # Get Whisper validator
        whisper_validator = get_whisper_validator("base")  # Use base model for balance of speed/accuracy

        # Transcribe the audio
        transcription_result = whisper_validator.transcribe_audio(audio_data, sample_rate=16000)

        if "error" in transcription_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {transcription_result['error']}"
            )

        # Prepare response data
        response_data = {
            "success": True,
            "filename": audio_file.filename,
            "transcription": transcription_result["transcription"],
            "segments": transcription_result["segments"],
            "transcription_info": {
                "transcription_time": transcription_result["transcription_time"],
                "detected_language": transcription_result.get("detected_language", "en"),
                "language_probability": transcription_result.get("language_probability", 1.0)
            }
        }

        # If original text provided, calculate accuracy metrics
        if original_text:
            validation_result = whisper_validator.validate_tts_output(
                original_text=original_text,
                audio_data=audio_data,
                sample_rate=16000
            )

            response_data.update({
                "original_text": original_text,
                "accuracy_metrics": validation_result.get("accuracy_metrics", {}),
                "word_alignment": validation_result.get("word_alignment", []),
                "quality_assessment": validation_result.get("quality_assessment", {})
            })

        # Add audio analysis
        try:
            from jabbertts.validation.audio_quality import AudioQualityValidator
            quality_validator = AudioQualityValidator()

            # Convert audio bytes to numpy array for analysis
            import soundfile as sf
            with io.BytesIO(audio_data) as audio_buffer:
                audio_array, sample_rate = sf.read(audio_buffer)

            # Ensure mono audio
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            # Analyze audio quality
            quality_metrics = quality_validator.analyze_audio(
                audio_array, sample_rate, rtf=0, inference_time=0
            )

            response_data["audio_analysis"] = {
                "duration": len(audio_array) / sample_rate,
                "sample_rate": sample_rate,
                "quality_metrics": quality_metrics.to_dict(),
                "quality_validation": quality_validator.validate_against_thresholds(quality_metrics)
            }

        except Exception as audio_error:
            logger.warning(f"Audio analysis failed: {audio_error}")
            response_data["audio_analysis"] = {"error": str(audio_error)}

        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug transcription failed: {str(e)}"
        )


@router.post("/debug/generate-and-transcribe")
async def debug_generate_and_transcribe(
    text: str = Form(...),
    voice: str = Form("alloy"),
    format: str = Form("wav"),
    speed: float = Form(1.0)
):
    """Generate TTS audio and immediately transcribe it for debugging.

    This endpoint provides end-to-end testing by:
    1. Generating TTS audio from input text
    2. Transcribing the generated audio with Whisper
    3. Comparing original vs transcribed text
    4. Providing detailed quality and accuracy metrics
    """
    try:
        logger.info(f"Debug generate-and-transcribe: {len(text)} chars, voice={voice}, format={format}")

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
        audio_data, audio_metadata = await audio_processor.process_audio(
            audio_array=tts_result["audio_data"],
            sample_rate=tts_result["sample_rate"],
            output_format=format,
            speed=speed
        )

        # Transcribe the generated audio
        whisper_validator = get_whisper_validator("base")
        validation_result = whisper_validator.validate_tts_output(
            original_text=text,
            audio_data=audio_data,
            sample_rate=tts_result["sample_rate"]
        )

        # Prepare comprehensive response
        response_data = {
            "success": True,
            "original_text": text,
            "generation_info": {
                "voice": voice,
                "format": format,
                "speed": speed,
                "rtf": tts_result.get("rtf", 0),
                "inference_time": tts_result.get("inference_time", 0),
                "audio_duration": tts_result.get("audio_duration", 0)
            },
            "transcription_result": validation_result,
            "audio_metadata": audio_metadata
        }

        # Add base64 encoded audio for playback
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        response_data["audio_data"] = f"data:audio/{format};base64,{audio_b64}"

        return JSONResponse(response_data)

    except Exception as e:
        logger.error(f"Debug generate-and-transcribe failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug generate-and-transcribe failed: {str(e)}"
        )


@router.post("/debug/audio-analysis")
async def debug_audio_analysis(
    text: str = Form(...),
    voice: str = Form("alloy"),
    format: str = Form("wav"),
    speed: float = Form(1.0),
    include_waveform: bool = Form(True),
    include_spectrogram: bool = Form(True),
    include_phonemes: bool = Form(True)
):
    """Generate comprehensive audio analysis with waveform visualization and phoneme alignment.

    This endpoint provides detailed audio analysis including:
    - Waveform visualization data
    - Spectrogram analysis
    - Phoneme alignment markers
    - Audio quality metrics
    - Processing pipeline breakdown
    """
    try:
        logger.info(f"Debug audio analysis: {len(text)} chars, voice={voice}, format={format}")

        # Generate TTS audio
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()

        # Get text preprocessor for phoneme analysis
        from jabbertts.inference.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor(use_phonemizer=True)

        # Generate speech
        tts_result = await inference_engine.generate_speech(
            text=text,
            voice=voice,
            speed=speed,
            response_format=format
        )

        # Process audio
        audio_data, audio_metadata = await audio_processor.process_audio(
            audio_array=tts_result["audio_data"],
            sample_rate=tts_result["sample_rate"],
            output_format=format,
            speed=speed
        )

        # Prepare analysis result
        analysis_result = {
            "success": True,
            "original_text": text,
            "generation_info": {
                "voice": voice,
                "format": format,
                "speed": speed,
                "rtf": tts_result.get("rtf", 0),
                "inference_time": tts_result.get("inference_time", 0),
                "audio_duration": tts_result.get("audio_duration", 0),
                "sample_rate": tts_result["sample_rate"]
            },
            "audio_metadata": audio_metadata
        }

        # Add phoneme analysis if requested
        if include_phonemes:
            try:
                phonemized_text = preprocessor.preprocess(text)
                phoneme_info = {
                    "original_text": text,
                    "phonemized_text": phonemized_text,
                    "phoneme_count": int(len(phonemized_text.split())),
                    "complexity_score": float(len([c for c in phonemized_text if c in "ˈˌːˑ"]) / len(phonemized_text) if phonemized_text else 0)
                }
                analysis_result["phoneme_analysis"] = phoneme_info
            except Exception as e:
                analysis_result["phoneme_analysis"] = {"error": str(e)}

        # Add waveform data if requested
        if include_waveform:
            try:
                raw_audio = tts_result["audio_data"]
                sample_rate = tts_result["sample_rate"]

                # Downsample for visualization (max 1000 points)
                downsample_factor = max(1, len(raw_audio) // 1000)
                waveform_data = [float(x) for x in raw_audio[::downsample_factor]]

                # Calculate time axis
                time_axis = [float(i * downsample_factor / sample_rate) for i in range(len(waveform_data))]

                analysis_result["waveform"] = {
                    "amplitude": waveform_data,
                    "time": time_axis,
                    "sample_rate": int(sample_rate),
                    "duration": float(len(raw_audio) / sample_rate),
                    "downsample_factor": int(downsample_factor)
                }
            except Exception as e:
                analysis_result["waveform"] = {"error": str(e)}

        # Add spectrogram data if requested
        if include_spectrogram:
            try:
                import numpy as np
                from scipy import signal

                raw_audio = tts_result["audio_data"]
                sample_rate = tts_result["sample_rate"]

                # Compute spectrogram
                frequencies, times, Sxx = signal.spectrogram(
                    raw_audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=512,
                    noverlap=256
                )

                # Convert to dB and downsample for visualization
                Sxx_db = 10 * np.log10(Sxx + 1e-10)

                # Downsample frequency axis (keep up to 100 frequency bins)
                freq_downsample = max(1, len(frequencies) // 100)
                frequencies_ds = [float(x) for x in frequencies[::freq_downsample]]
                Sxx_db_ds = Sxx_db[::freq_downsample, :]

                # Downsample time axis (keep up to 200 time bins)
                time_downsample = max(1, Sxx_db_ds.shape[1] // 200)
                times_ds = [float(x) for x in times[::time_downsample]]
                Sxx_db_ds = Sxx_db_ds[:, ::time_downsample]

                analysis_result["spectrogram"] = {
                    "frequencies": frequencies_ds,
                    "times": times_ds,
                    "magnitude_db": [[float(x) for x in row] for row in Sxx_db_ds.tolist()],
                    "freq_downsample": int(freq_downsample),
                    "time_downsample": int(time_downsample)
                }
            except Exception as e:
                analysis_result["spectrogram"] = {"error": str(e)}

        # Add audio quality analysis
        try:
            from jabbertts.validation.audio_quality import AudioQualityValidator
            quality_validator = AudioQualityValidator()

            # Analyze raw audio quality
            quality_metrics = quality_validator.analyze_audio(
                tts_result["audio_data"],
                tts_result["sample_rate"],
                tts_result.get("rtf", 0),
                tts_result.get("inference_time", 0)
            )

            # Convert validation results to JSON-serializable format
            validation_results = quality_validator.validate_against_thresholds(quality_metrics)
            validation_json = {k: bool(v) for k, v in validation_results.items()}

            analysis_result["quality_analysis"] = {
                "metrics": quality_metrics.to_dict(),
                "validation": validation_json,
                "overall_score": float(quality_metrics.overall_quality)
            }

        except Exception as e:
            analysis_result["quality_analysis"] = {"error": str(e)}

        # Add base64 encoded audio for playback
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        analysis_result["audio_data"] = f"data:audio/{format};base64,{audio_b64}"

        return JSONResponse(analysis_result)

    except Exception as e:
        logger.error(f"Debug audio analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug audio analysis failed: {str(e)}"
        )
