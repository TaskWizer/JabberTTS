"""Inference Engine for JabberTTS.

This module provides the main inference engine that coordinates
text preprocessing, model inference, and audio generation.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
import numpy as np

from jabbertts.config import get_settings
from jabbertts.models.manager import get_model_manager
from .preprocessing import TextPreprocessor
from jabbertts.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Main inference engine for TTS generation.
    
    Coordinates text preprocessing, model inference, and performance monitoring
    to generate high-quality speech from text input.
    """
    
    def __init__(self):
        """Initialize the inference engine."""
        self.settings = get_settings()
        self.model_manager = get_model_manager()
        self.preprocessor = TextPreprocessor(use_phonemizer=True)
        self.performance_stats = {
            "total_requests": 0,
            "total_inference_time": 0.0,
            "total_audio_duration": 0.0,
            "average_rtf": 0.0
        }
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate speech from text asynchronously.
        
        Args:
            text: Input text to convert to speech
            voice: Voice identifier to use
            speed: Speech speed multiplier (0.25-4.0)
            response_format: Audio format for output
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing audio data and metadata
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If inference fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting TTS generation for {len(text)} characters")
            
            # Validate input parameters
            self._validate_input(text, voice, speed, response_format)
            
            # Preprocess text
            processed_text = await self._preprocess_text(text)
            logger.debug(f"Preprocessed text: '{processed_text[:100]}...'")
            
            # Load model if needed
            model = await self._ensure_model_loaded()
            
            # Generate speech using the model
            audio_data = await self._generate_audio(model, processed_text, voice, speed)

            # Calculate performance metrics (using raw audio before processing)
            inference_time = time.time() - start_time
            raw_audio_duration = len(audio_data) / model.get_sample_rate()
            rtf = inference_time / raw_audio_duration if raw_audio_duration > 0 else 0
            
            # Update performance statistics
            self._update_performance_stats(inference_time, raw_audio_duration, rtf)

            # Record metrics
            metrics_collector = get_metrics_collector()
            metrics_collector.record_request(
                duration=inference_time,
                success=True,
                rtf=rtf,
                audio_duration=raw_audio_duration,
                text_length=len(text),
                voice=voice,
                format=response_format
            )

            logger.info(f"TTS generation completed in {inference_time:.2f}s (RTF: {rtf:.3f})")

            return {
                "audio_data": audio_data,
                "sample_rate": model.get_sample_rate(),
                "duration": raw_audio_duration,  # Raw duration for RTF calculation
                "inference_time": inference_time,
                "rtf": rtf,
                "voice": voice,
                "speed": speed,
                "format": response_format,
                "text_length": len(text),
                "processed_text_length": len(processed_text)
            }
            
        except Exception as e:
            error_time = time.time() - start_time

            # Record failed request metrics
            metrics_collector = get_metrics_collector()
            metrics_collector.record_request(
                duration=error_time,
                success=False,
                text_length=len(text),
                voice=voice,
                format=response_format
            )

            logger.error(f"TTS generation failed after {error_time:.2f}s: {e}")
            raise RuntimeError(f"Speech generation failed: {e}") from e
    
    def generate_speech_sync(
        self,
        text: str,
        voice: str = "alloy", 
        speed: float = 1.0,
        response_format: str = "mp3",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate speech synchronously (wrapper for async method).
        
        Args:
            text: Input text to convert to speech
            voice: Voice identifier to use
            speed: Speech speed multiplier
            response_format: Audio format for output
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing audio data and metadata
        """
        try:
            # Get or create event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate_speech(text, voice, speed, response_format, **kwargs)
        )
    
    async def _preprocess_text(self, text: str) -> str:
        """Preprocess text asynchronously.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Run preprocessing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.preprocessor.preprocess, 
            text
        )
    
    async def _ensure_model_loaded(self):
        """Ensure TTS model is loaded.
        
        Returns:
            Loaded TTS model
        """
        model = self.model_manager.get_current_model()
        
        if model is None or not model.is_loaded:
            logger.info("Loading TTS model...")
            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                self.model_manager.load_model
            )
        
        return model
    
    async def _generate_audio(self, model, text: str, voice: str, speed: float) -> np.ndarray:
        """Generate audio using the TTS model.
        
        Args:
            model: TTS model instance
            text: Preprocessed text
            voice: Voice identifier
            speed: Speech speed
            
        Returns:
            Generated audio data
        """
        # Run model inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            model.generate_speech,
            text,
            voice,
            speed
        )
    
    def _validate_input(self, text: str, voice: str, speed: float, response_format: str) -> None:
        """Validate input parameters.
        
        Args:
            text: Input text
            voice: Voice identifier
            speed: Speech speed
            response_format: Audio format
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > self.settings.max_text_length:
            raise ValueError(f"Text exceeds maximum length of {self.settings.max_text_length} characters")
        
        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        valid_formats = ["mp3", "wav", "flac", "opus", "aac", "pcm"]
        if response_format not in valid_formats:
            raise ValueError(f"Invalid format '{response_format}'. Valid formats: {valid_formats}")
    
    def _update_performance_stats(self, inference_time: float, audio_duration: float, rtf: float) -> None:
        """Update performance statistics.
        
        Args:
            inference_time: Time taken for inference
            audio_duration: Duration of generated audio
            rtf: Real-time factor
        """
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_inference_time"] += inference_time
        self.performance_stats["total_audio_duration"] += audio_duration
        
        # Calculate running average RTF
        total_requests = self.performance_stats["total_requests"]
        total_inf_time = self.performance_stats["total_inference_time"]
        total_audio_time = self.performance_stats["total_audio_duration"]
        
        if total_audio_time > 0:
            self.performance_stats["average_rtf"] = total_inf_time / total_audio_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            **self.performance_stats,
            "model_status": self.model_manager.get_model_status(),
            "preprocessing_info": self.preprocessor.get_preprocessing_info()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the inference engine.
        
        Returns:
            Dictionary with health information
        """
        model = self.model_manager.get_current_model()
        
        return {
            "status": "healthy" if model and model.is_loaded else "degraded",
            "model_loaded": model.is_loaded if model else False,
            "model_name": self.settings.model_name,
            "preprocessor_ready": True,
            "performance": {
                "average_rtf": self.performance_stats["average_rtf"],
                "total_requests": self.performance_stats["total_requests"]
            }
        }
    
    async def warmup(self, num_runs: int = 5) -> None:
        """Warm up the inference engine with test generations.

        This helps ensure models are loaded, compiled, and ready for production use.
        Uses progressive warmup to achieve optimal RTF performance.

        Args:
            num_runs: Number of warmup runs to perform
        """
        try:
            logger.info(f"Warming up inference engine with {num_runs} runs...")

            # Progressive warmup with different text complexities
            warmup_texts = [
                "Hi",  # Very short - triggers basic compilation
                "Hello world",  # Short
                "Testing model compilation and optimization.",  # Medium
                "This is a comprehensive warmup test for the TTS system.",  # Long
                "Ready for production use with optimal performance."  # Final
            ]

            rtfs = []
            voices = ["alloy", "echo", "nova"]  # Test multiple voices for cache warming

            for i in range(num_runs):
                text = warmup_texts[i % len(warmup_texts)]
                voice = voices[i % len(voices)]

                logger.debug(f"Warmup run {i+1}/{num_runs}: '{text[:30]}...' (voice: {voice})")

                result = await self.generate_speech(
                    text=text,
                    voice=voice,
                    speed=1.0
                )
                rtfs.append(result['rtf'])
                logger.debug(f"Warmup run {i+1}: RTF = {result['rtf']:.3f}")

            avg_rtf = sum(rtfs) / len(rtfs)
            best_rtf = min(rtfs)
            worst_rtf = max(rtfs)
            final_rtf = rtfs[-1]  # Last run should be most optimized

            logger.info(f"Warmup completed successfully")
            logger.info(f"RTF Performance - Avg: {avg_rtf:.3f}, Best: {best_rtf:.3f}, Final: {final_rtf:.3f}")

            if final_rtf < 0.5:
                logger.info("✓ Performance target achieved (Final RTF < 0.5)")
            else:
                logger.warning(f"⚠ Performance target not met (Final RTF {final_rtf:.3f} >= 0.5)")

            # Additional optimization attempt if performance is poor
            if final_rtf > 0.6:
                logger.info("Attempting additional optimization warmup...")
                await self._additional_warmup()

        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    async def _additional_warmup(self) -> None:
        """Perform additional warmup for poor-performing models."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Additional warmup runs with the most common use case
            for i in range(3):
                result = await self.generate_speech(
                    text="Hello, this is a test message.",
                    voice="alloy",
                    speed=1.0
                )
                logger.debug(f"Additional warmup {i+1}: RTF = {result['rtf']:.3f}")

        except Exception as e:
            logger.debug(f"Additional warmup failed: {e}")


# Global inference engine instance
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get the global inference engine instance.
    
    Returns:
        Global InferenceEngine instance
    """
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
