"""Base TTS Model Interface.

This module defines the abstract base class for all TTS models in JabberTTS,
ensuring consistent interface and functionality across different model implementations.
"""

import abc
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum
import torch
import numpy as np


class ModelCapability(Enum):
    """Model capability flags."""
    FAST_INFERENCE = "fast_inference"
    HIGH_QUALITY = "high_quality"
    LONG_TEXT = "long_text"
    VOICE_CLONING = "voice_cloning"
    EMOTION_CONTROL = "emotion_control"
    MULTILINGUAL = "multilingual"


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""
    pass


class InferenceError(ModelError):
    """Exception raised during inference."""
    pass


class UnsupportedInputError(ModelError):
    """Exception raised for unsupported input parameters."""
    pass


class TTSModelInterface(abc.ABC):
    """Enhanced abstract interface for TTS models with standardized methods."""

    @abc.abstractmethod
    def generate_speech(self, text: str, voice: str, **kwargs) -> np.ndarray:
        """Generate speech audio from text."""
        pass

    @abc.abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of generated audio."""
        pass

    @abc.abstractmethod
    def get_supported_voices(self) -> List[str]:
        """Get list of supported voice identifiers."""
        pass

    @abc.abstractmethod
    def validate_input(self, text: str) -> bool:
        """Validate if the input text is supported by this model."""
        pass

    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        pass

    @abc.abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """Get list of model capabilities."""
        pass

    @abc.abstractmethod
    def get_max_text_length(self) -> int:
        """Get maximum supported text length."""
        pass

    @abc.abstractmethod
    def estimate_inference_time(self, text: str) -> float:
        """Estimate inference time for given text."""
        pass


class BaseTTSModel(TTSModelInterface):
    """Abstract base class for TTS models.
    
    This class defines the interface that all TTS models must implement
    to be compatible with the JabberTTS system.
    """
    
    def __init__(self, model_path: Union[str, Path], device: str = "cpu"):
        """Initialize the TTS model.
        
        Args:
            model_path: Path to the model files
            device: Device to run the model on (cpu/cuda)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.is_loaded = False
        
    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the model into memory.
        
        This method should load all necessary model components
        and set self.is_loaded = True when complete.
        """
        pass
    
    @abc.abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory.
        
        This method should free up memory used by the model
        and set self.is_loaded = False.
        """
        pass
    
    @abc.abstractmethod
    def generate_speech(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """Generate speech audio from text.

        Args:
            text: Input text to convert to speech
            voice: Voice identifier to use
            speed: Speech speed multiplier (0.25-4.0)
            **kwargs: Additional model-specific parameters

        Returns:
            Audio data as numpy array (float32, sample_rate defined by model)

        Raises:
            ModelLoadError: If model is not loaded
            InferenceError: If inference fails
            UnsupportedInputError: If parameters are invalid
        """
        pass
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice identifiers.

        Returns:
            List of voice identifiers supported by this model
        """
        return self.get_supported_voices()

    def get_supported_voices(self) -> List[str]:
        """Get list of supported voice identifiers.

        Returns:
            List of voice identifiers supported by this model
        """
        return ["default"]
    
    @abc.abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of generated audio.
        
        Returns:
            Sample rate in Hz
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata.

        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.__class__.__name__,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "sample_rate": self.get_sample_rate(),
            "voices": len(self.get_supported_voices()),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "max_text_length": self.get_max_text_length()
        }

    def get_capabilities(self) -> List[ModelCapability]:
        """Get list of model capabilities.

        Returns:
            List of ModelCapability enums
        """
        return []

    def get_max_text_length(self) -> int:
        """Get maximum supported text length.

        Returns:
            Maximum text length in characters
        """
        return 4096

    def validate_input(self, text: str) -> bool:
        """Validate if the input text is supported by this model.

        Args:
            text: Input text to validate

        Returns:
            True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False
        if len(text) > self.get_max_text_length():
            return False
        return True

    def estimate_inference_time(self, text: str) -> float:
        """Estimate inference time for given text.

        Args:
            text: Input text

        Returns:
            Estimated inference time in seconds
        """
        # Basic estimation: ~0.1 seconds per 10 characters
        return len(text) * 0.01
    
    def validate_parameters(self, text: str, voice: str, speed: float) -> None:
        """Validate input parameters.
        
        Args:
            text: Input text
            voice: Voice identifier
            speed: Speech speed
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > 4096:
            raise ValueError("Text exceeds maximum length of 4096 characters")
        
        if not 0.25 <= speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")
        
        available_voices = self.get_available_voices()
        if voice not in available_voices:
            raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")
    
    def ensure_loaded(self) -> None:
        """Ensure the model is loaded.
        
        Raises:
            RuntimeError: If model failed to load
        """
        if not self.is_loaded:
            self.load_model()
        
        if not self.is_loaded:
            raise RuntimeError("Failed to load model")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of the model.
        
        Returns:
            Dictionary with memory usage information in MB
        """
        if not self.is_loaded or self.model is None:
            return {"model_memory": 0.0, "total_memory": 0.0}
        
        model_memory = 0.0
        if hasattr(self.model, 'parameters'):
            for param in self.model.parameters():
                model_memory += param.numel() * param.element_size()
        
        # Convert to MB
        model_memory_mb = model_memory / (1024 * 1024)
        
        # Get total memory if using CUDA
        total_memory_mb = 0.0
        if self.device.startswith('cuda') and torch.cuda.is_available():
            total_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return {
            "model_memory": model_memory_mb,
            "total_memory": total_memory_mb
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.ensure_loaded()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Optionally unload model on exit
        pass
