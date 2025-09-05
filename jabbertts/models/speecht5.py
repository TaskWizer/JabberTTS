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
            
            # Load speaker embeddings dataset with trust_remote_code=True
            try:
                embeddings_dataset = datasets.load_dataset(
                    "Matthijs/cmu-arctic-xvectors",
                    split="validation",
                    trust_remote_code=True
                )
                self.speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
            except Exception as e:
                logger.warning(f"Could not load speaker embeddings dataset: {e}")
                # Create a dummy speaker embedding as fallback
                self.speaker_embeddings = torch.randn(1, 512)  # Standard xvector size
            
            self.is_loaded = True
            logger.info("SpeechT5 model loaded successfully")
            
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
        self.is_loaded = False
        
        # Clear CUDA cache if using GPU
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SpeechT5 model unloaded")
    
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
            
            # Preprocess text
            inputs = self.processor(text=text, return_tensors="pt")
            
            # Move inputs to device
            input_ids = inputs["input_ids"].to(self.device)
            
            # Get speaker embeddings for the requested voice
            speaker_embeddings = self._get_speaker_embeddings(voice).to(self.device)
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    input_ids, 
                    speaker_embeddings, 
                    vocoder=self.vocoder
                )
            
            # Convert to numpy
            audio = speech.cpu().numpy()
            
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
        """Get speaker embeddings for a voice.
        
        Args:
            voice: Voice identifier
            
        Returns:
            Speaker embeddings tensor
        """
        # For now, use the same embeddings for all voices
        # In a full implementation, we'd have different embeddings per voice
        if self.speaker_embeddings is None:
            raise RuntimeError("Speaker embeddings not loaded")
        
        return self.speaker_embeddings
    
    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio speed using simple resampling.
        
        Args:
            audio: Input audio array
            speed: Speed multiplier
            
        Returns:
            Speed-adjusted audio array
        """
        try:
            import librosa
            # Use librosa for time stretching
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
