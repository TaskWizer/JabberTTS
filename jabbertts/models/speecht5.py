"""SpeechT5 TTS Model Implementation.

This module provides a SpeechT5-based TTS model implementation as an initial
working model for JabberTTS. SpeechT5 is readily available and doesn't require
special permissions, making it ideal for testing and development.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from .base import BaseTTSModel

logger = logging.getLogger(__name__)


class SpeechT5Model(BaseTTSModel):
    """SpeechT5 TTS model implementation.
    
    This implementation uses Microsoft's SpeechT5 model for text-to-speech
    generation. It serves as a working baseline while the infrastructure
    is being developed for more advanced models.
    """
    
    DESCRIPTION = "Microsoft SpeechT5 TTS model - baseline implementation"
    SAMPLE_RATE = 16000  # SpeechT5 uses 16kHz
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        """Initialize SpeechT5 model.
        
        Args:
            model_path: Path to model files (not used for SpeechT5 as it downloads automatically)
            device: Device to run on
        """
        super().__init__(model_path, device)
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self._speaker_embeddings_cache = {}  # Cache for speaker embeddings
        
    def load_model(self) -> None:
        """Load SpeechT5 model components."""
        try:
            logger.info("Loading SpeechT5 model...")
            
            # Import transformers components
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import datasets
            
            # Load processor and model
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move to device
            self.model = self.model.to(self.device)
            self.vocoder = self.vocoder.to(self.device)

            # Optimize models for inference
            self.model.eval()
            self.vocoder.eval()

            # Additional optimizations for better performance
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

            # Set optimal number of threads for CPU inference
            if self.device == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))  # Optimal for most CPUs

            # Try to compile models for better performance (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile'):
                    logger.info("Compiling models for better performance...")
                    # Use more aggressive compilation for better RTF
                    self.model = torch.compile(self.model, mode='max-autotune', fullgraph=True)
                    self.vocoder = torch.compile(self.vocoder, mode='max-autotune', fullgraph=True)
                    logger.info("Model compilation successful with max-autotune")

                    # Note: Compilation warmup will be performed after speaker embeddings are loaded
            except Exception as e:
                logger.warning(f"Model compilation failed (this is okay): {e}")
                # Fallback to basic optimization
                try:
                    self.model.eval()
                    self.vocoder.eval()
                    with torch.no_grad():
                        # Set inference mode for better performance
                        torch.set_grad_enabled(False)
                except Exception:
                    pass
            
            # Load speaker embeddings - use fallback approach due to trust_remote_code deprecation
            try:
                # Try to load without trust_remote_code first
                embeddings_dataset = datasets.load_dataset(
                    "Matthijs/cmu-arctic-xvectors",
                    split="validation"
                )
                self.speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
                logger.info("Successfully loaded speaker embeddings from dataset")
            except Exception as e:
                logger.warning(f"Could not load speaker embeddings dataset: {e}")
                # Create multiple speaker embeddings for different voices
                self.speaker_embeddings = self._create_voice_embeddings()
                logger.info("Created voice embeddings for OpenAI-compatible voices")
            
            self.is_loaded = True
            logger.info("SpeechT5 model loaded successfully")

            # Perform compilation warmup now that everything is loaded
            if hasattr(torch, 'compile') and hasattr(self.model, '__wrapped__'):
                try:
                    logger.info("Performing compilation warmup...")
                    self._perform_compilation_warmup()
                    logger.info("Compilation warmup completed")
                except Exception as e:
                    logger.warning(f"Compilation warmup failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load SpeechT5 model: {e}")
            self.is_loaded = False
            raise
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self._speaker_embeddings_cache.clear()  # Clear cache
        self.is_loaded = False
        
        # Clear CUDA cache if using GPU
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SpeechT5 model unloaded")

    def _perform_compilation_warmup(self) -> None:
        """Perform warmup to trigger torch compilation and optimize performance."""
        try:
            # Warmup with different text lengths to trigger compilation paths
            warmup_texts = [
                "Hi",  # Very short
                "Hello world test",  # Medium
                "This is a longer text for compilation warmup testing purposes"  # Long
            ]

            for i, text in enumerate(warmup_texts):
                logger.debug(f"Compilation warmup {i+1}/3: '{text[:30]}...'")

                # Preprocess text
                inputs = self.processor(text=text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)

                # Get default speaker embeddings
                speaker_embeddings = self._get_speaker_embeddings("default").to(self.device)

                # Run inference to trigger compilation
                with torch.no_grad():
                    _ = self.model.generate_speech(
                        input_ids,
                        speaker_embeddings,
                        vocoder=self.vocoder
                    )

                logger.debug(f"Compilation warmup {i+1}/3 completed")

        except Exception as e:
            logger.warning(f"Compilation warmup failed: {e}")
            # This is not critical, continue without warmup
    
    def generate_speech(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """Generate speech from text using SpeechT5.
        
        Args:
            text: Input text
            voice: Voice identifier (mapped to speaker embeddings)
            speed: Speech speed (applied via post-processing)
            **kwargs: Additional parameters
            
        Returns:
            Audio data as numpy array (float32, 16kHz)
        """
        self.ensure_loaded()
        self.validate_parameters(text, voice, speed)
        
        try:
            logger.debug(f"Generating speech for text: '{text[:50]}...'")

            # Optimize inference with torch settings
            with torch.inference_mode():  # More efficient than no_grad for inference
                # CRITICAL FIX: SpeechT5 expects raw text, not phonemes
                # The processor has its own tokenization that conflicts with eSpeak phonemization
                # Use raw text directly without any phoneme preprocessing
                inputs = self.processor(text=text, return_tensors="pt")

                # Move inputs to device efficiently
                input_ids = inputs["input_ids"].to(self.device, non_blocking=True)

                # Get speaker embeddings for the requested voice (cached)
                speaker_embeddings = self._get_speaker_embeddings(voice).to(self.device, non_blocking=True)

                # Generate speech with optimized settings
                speech = self.model.generate_speech(
                    input_ids,
                    speaker_embeddings,
                    vocoder=self.vocoder
                )

                # Convert to numpy efficiently
                audio = speech.detach().cpu().numpy()

            # Apply speed adjustment if needed
            if speed != 1.0:
                audio = self._adjust_speed(audio, speed)

            logger.debug(f"Generated audio: {audio.shape} samples at {self.SAMPLE_RATE}Hz")
            return audio
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Failed to generate speech: {e}") from e
    
    def get_available_voices(self) -> List[str]:
        """Get available voice identifiers.
        
        Returns:
            List of available voice names
        """
        # SpeechT5 uses speaker embeddings, we'll provide some default mappings
        return [
            "default",
            "alloy",    # Map to default for OpenAI compatibility
            "echo",
            "fable", 
            "onyx",
            "nova",
            "shimmer"
        ]
    
    def get_sample_rate(self) -> int:
        """Get sample rate of generated audio.
        
        Returns:
            Sample rate in Hz
        """
        return self.SAMPLE_RATE
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            "name": "SpeechT5",
            "description": self.DESCRIPTION,
            "sample_rate": self.SAMPLE_RATE,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "model_size": "~300MB",
            "languages": ["en"],
            "voices": len(self.get_available_voices())
        }
    
    def _get_speaker_embeddings(self, voice: str) -> torch.Tensor:
        """Get speaker embeddings for a voice with caching for performance.

        Args:
            voice: Voice identifier (OpenAI-compatible)

        Returns:
            Speaker embeddings tensor for the specified voice
        """
        # Check cache first for performance
        if voice in self._speaker_embeddings_cache:
            return self._speaker_embeddings_cache[voice]

        if self.speaker_embeddings is None:
            raise RuntimeError("Speaker embeddings not loaded")

        # Map OpenAI voice names to our embeddings
        voice_mapping = {
            "alloy": 0,      # Neutral, balanced voice
            "echo": 1,       # Clear, crisp voice
            "fable": 2,      # Warm, storytelling voice
            "onyx": 3,       # Deep, authoritative voice
            "nova": 4,       # Bright, energetic voice
            "shimmer": 5     # Soft, gentle voice
        }

        voice_index = voice_mapping.get(voice.lower(), 0)  # Default to alloy

        if isinstance(self.speaker_embeddings, dict):
            embeddings = self.speaker_embeddings[voice_index]
        else:
            # Fallback for single embedding
            embeddings = self.speaker_embeddings

        # Cache the embeddings for future use
        self._speaker_embeddings_cache[voice] = embeddings
        return embeddings

    def _create_voice_embeddings(self) -> Dict[int, torch.Tensor]:
        """Create speaker embeddings for different OpenAI-compatible voices.

        Returns:
            Dictionary mapping voice indices to speaker embedding tensors
        """
        voice_embeddings = {}

        # Voice characteristics (these create different voice timbres)
        voice_configs = [
            {"name": "alloy", "base_scale": 0.1, "low": 1.0, "mid": 0.9, "high": 0.8},      # Neutral
            {"name": "echo", "base_scale": 0.12, "low": 0.8, "mid": 1.1, "high": 1.0},     # Clear/crisp
            {"name": "fable", "base_scale": 0.09, "low": 1.2, "mid": 0.8, "high": 0.7},    # Warm
            {"name": "onyx", "base_scale": 0.15, "low": 1.4, "mid": 0.9, "high": 0.6},     # Deep
            {"name": "nova", "base_scale": 0.11, "low": 0.7, "mid": 1.0, "high": 1.2},     # Bright
            {"name": "shimmer", "base_scale": 0.08, "low": 0.9, "mid": 0.8, "high": 0.9}   # Soft
        ]

        for i, config in enumerate(voice_configs):
            # Create base embedding
            embedding = torch.randn(1, 512) * config["base_scale"]

            # Apply voice-specific frequency characteristics
            embedding[:, :128] *= config["low"]      # Low frequency characteristics
            embedding[:, 128:256] *= config["mid"]   # Mid frequency characteristics
            embedding[:, 256:] *= config["high"]     # High frequency characteristics

            # Add some deterministic variation based on voice index
            torch.manual_seed(42 + i)  # Reproducible but different per voice
            variation = torch.randn(1, 512) * 0.02
            embedding += variation

            # Normalize to unit length
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            voice_embeddings[i] = embedding

        return voice_embeddings

    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio speed using advanced time-stretching algorithms.

        Args:
            audio: Input audio array
            speed: Speed multiplier

        Returns:
            Speed-adjusted audio array with preserved quality
        """
        try:
            from jabbertts.audio.advanced_speed_control import adjust_audio_speed
            return adjust_audio_speed(
                audio=audio,
                speed_factor=speed,
                sample_rate=self.get_sample_rate(),
                preserve_pitch=True
            )
        except ImportError:
            logger.warning("Advanced speed control not available, using fallback")
            try:
                import librosa
                return librosa.effects.time_stretch(audio, rate=speed)
            except ImportError:
                logger.warning("librosa not available, speed adjustment disabled")
                return audio
        except Exception as e:
            logger.warning(f"Speed adjustment failed: {e}")
            return audio
    
    @classmethod
    def validate_files(cls, model_path: Path) -> bool:
        """Validate model files (not needed for SpeechT5 as it auto-downloads).
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Always True for SpeechT5
        """
        return True
