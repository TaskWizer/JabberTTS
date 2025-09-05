"""Model Manager for JabberTTS.

This module provides centralized model management including loading,
caching, and switching between different TTS models.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Type, Any
import hashlib
import json

from jabbertts.config import get_settings
from .base import BaseTTSModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized manager for TTS models.
    
    Handles model loading, caching, validation, and switching between
    different TTS model implementations.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.settings = get_settings()
        self.current_model: Optional[BaseTTSModel] = None
        self.model_cache: Dict[str, BaseTTSModel] = {}
        self.model_registry: Dict[str, Type[BaseTTSModel]] = {}
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register default model implementations."""
        try:
            # Import and register available models
            from .speecht5 import SpeechT5Model
            self.register_model("speecht5", SpeechT5Model)
            logger.info("Registered SpeechT5 model")
        except ImportError as e:
            logger.warning(f"Could not register SpeechT5 model: {e}")
        
        try:
            from .openaudio import OpenAudioS1MiniModel
            self.register_model("openaudio-s1-mini", OpenAudioS1MiniModel)
            logger.info("Registered OpenAudio S1-mini model")
        except ImportError as e:
            logger.warning(f"Could not register OpenAudio S1-mini model: {e}")
    
    def register_model(self, name: str, model_class: Type[BaseTTSModel]) -> None:
        """Register a model implementation.
        
        Args:
            name: Model identifier
            model_class: Model class implementing BaseTTSModel
        """
        self.model_registry[name] = model_class
        logger.info(f"Registered model: {name}")
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        models = {}
        for name, model_class in self.model_registry.items():
            try:
                # Try to get model info without loading
                models[name] = getattr(model_class, 'DESCRIPTION', f"{name} TTS model")
            except Exception as e:
                logger.warning(f"Could not get info for model {name}: {e}")
                models[name] = f"{name} (unavailable)"
        
        return models
    
    def load_model(self, model_name: Optional[str] = None, force_reload: bool = False) -> BaseTTSModel:
        """Load a TTS model.
        
        Args:
            model_name: Name of the model to load (defaults to configured model)
            force_reload: Force reload even if model is cached
            
        Returns:
            Loaded TTS model instance
            
        Raises:
            ValueError: If model is not available
            RuntimeError: If model loading fails
        """
        if model_name is None:
            model_name = self.settings.model_name
        
        # Check if model is already loaded and cached
        if not force_reload and model_name in self.model_cache:
            cached_model = self.model_cache[model_name]
            if cached_model.is_loaded:
                logger.info(f"Using cached model: {model_name}")
                self.current_model = cached_model
                return cached_model
        
        # Check if model is registered
        if model_name not in self.model_registry:
            available = list(self.model_registry.keys())
            raise ValueError(f"Model '{model_name}' not available. Available models: {available}")
        
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            # Get model class and instantiate
            model_class = self.model_registry[model_name]
            model_path = self._get_model_path(model_name)
            
            # Create model instance
            model = model_class(model_path=model_path, device=self._get_device())
            
            # Load the model
            model.load_model()
            
            # Cache the model
            self.model_cache[model_name] = model
            self.current_model = model
            
            load_time = time.time() - start_time
            logger.info(f"Model '{model_name}' loaded successfully in {load_time:.2f}s")
            
            # Log model info
            model_info = model.get_model_info()
            logger.info(f"Model info: {model_info}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def get_current_model(self) -> Optional[BaseTTSModel]:
        """Get the currently loaded model.
        
        Returns:
            Current model instance or None if no model is loaded
        """
        return self.current_model
    
    def unload_model(self, model_name: Optional[str] = None) -> None:
        """Unload a model from memory.
        
        Args:
            model_name: Name of model to unload (defaults to current model)
        """
        if model_name is None:
            if self.current_model is not None:
                model_name = self._get_model_name(self.current_model)
            else:
                return
        
        if model_name in self.model_cache:
            model = self.model_cache[model_name]
            model.unload_model()
            del self.model_cache[model_name]
            logger.info(f"Unloaded model: {model_name}")
            
            if self.current_model == model:
                self.current_model = None
    
    def unload_all_models(self) -> None:
        """Unload all cached models."""
        for model_name in list(self.model_cache.keys()):
            self.unload_model(model_name)
        
        self.current_model = None
        logger.info("Unloaded all models")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models.
        
        Returns:
            Dictionary with model status information
        """
        status = {
            "current_model": self._get_model_name(self.current_model) if self.current_model else None,
            "loaded_models": list(self.model_cache.keys()),
            "available_models": self.get_available_models(),
            "memory_usage": {}
        }
        
        # Get memory usage for loaded models
        for name, model in self.model_cache.items():
            if model.is_loaded:
                status["memory_usage"][name] = model.get_memory_usage()
        
        return status
    
    def validate_model_files(self, model_name: str) -> bool:
        """Validate that model files exist and are valid.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model files are valid
        """
        try:
            # Check if model is registered
            if model_name not in self.model_registry:
                logger.warning(f"Model '{model_name}' not registered")
                return False

            model_class = self.model_registry[model_name]

            # Use model-specific validation if available
            if hasattr(model_class, 'validate_files'):
                model_path = self._get_model_path(model_name)
                return model_class.validate_files(model_path)

            # Default validation: check if model path exists
            model_path = self._get_model_path(model_name)
            if not model_path.exists():
                logger.warning(f"Model path does not exist: {model_path}")
                return False

            return True

        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            return False
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get the path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model directory
        """
        if self.settings.model_path:
            return Path(self.settings.model_path)
        
        # Default model paths
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        
        return models_dir / model_name
    
    def _get_device(self) -> str:
        """Get the device to use for model inference.
        
        Returns:
            Device string (cpu/cuda)
        """
        # For now, always use CPU as per requirements
        return "cpu"
    
    def _get_model_name(self, model: BaseTTSModel) -> Optional[str]:
        """Get the name of a model instance.
        
        Args:
            model: Model instance
            
        Returns:
            Model name or None if not found
        """
        for name, cached_model in self.model_cache.items():
            if cached_model == model:
                return name
        return None


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance.
    
    Returns:
        Global ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
