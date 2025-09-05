"""Model Manager for JabberTTS.

This module provides centralized model management including loading,
caching, and switching between different TTS models with intelligent
selection and fallback mechanisms.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Type, Any, List, Tuple
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from jabbertts.config import get_settings
from .base import BaseTTSModel, ModelCapability, ModelError, ModelLoadError, InferenceError

logger = logging.getLogger(__name__)


class ModelSelectionStrategy(Enum):
    """Model selection strategies."""
    AUTO = "auto"
    FAST = "fast"
    QUALITY = "quality"
    BALANCED = "balanced"


@dataclass
class ModelSelectionCriteria:
    """Criteria for intelligent model selection."""
    text_length: int
    quality_requirement: str = "medium"  # low, medium, high
    performance_requirement: str = "medium"  # low, medium, high
    max_rtf: float = 0.5
    voice: str = "alloy"
    strategy: ModelSelectionStrategy = ModelSelectionStrategy.AUTO


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    average_rtf: float = 0.0
    success_rate: float = 1.0
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    total_requests: int = 0


class ModelManager:
    """Centralized manager for TTS models with intelligent selection and fallback.

    Handles model loading, caching, validation, switching, and intelligent
    selection between different TTS model implementations.
    """

    def __init__(self):
        """Initialize the model manager."""
        self.settings = get_settings()
        self.current_model: Optional[BaseTTSModel] = None
        self.model_cache: Dict[str, BaseTTSModel] = {}
        self.model_registry: Dict[str, Type[BaseTTSModel]] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_priority: List[str] = []  # Priority order for fallback
        self.circuit_breaker_threshold = 3  # Failures before temporary disable
        self.circuit_breaker_timeout = 300  # 5 minutes
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register default model implementations with priority order."""
        try:
            # Import and register models in priority order (best first)

            # Priority 1: OpenAudio S1-mini (if available)
            try:
                from .openaudio import OpenAudioS1MiniModel
                self.register_model("openaudio-s1-mini", OpenAudioS1MiniModel, priority=1)
                logger.info("Registered OpenAudio S1-mini model")
            except ImportError as e:
                logger.debug(f"OpenAudio S1-mini model not available: {e}")

            # Priority 2: Coqui TTS VITS (if available)
            try:
                from .coqui_vits import CoquiVITSModel
                self.register_model("coqui-vits", CoquiVITSModel, priority=2)
                logger.info("Registered Coqui TTS VITS model")
            except ImportError as e:
                logger.debug(f"Coqui TTS VITS model not available: {e}")

            # Priority 3: SpeechT5 (fallback)
            from .speecht5 import SpeechT5Model
            self.register_model("speecht5", SpeechT5Model, priority=3)
            logger.info("Registered SpeechT5 model")

        except Exception as e:
            logger.error(f"Failed to register default models: {e}")

    def register_model(self, name: str, model_class: Type[BaseTTSModel], priority: int = 999) -> None:
        """Register a model implementation with priority.

        Args:
            name: Model identifier
            model_class: Model class implementing BaseTTSModel
            priority: Priority for model selection (lower = higher priority)
        """
        self.model_registry[name] = model_class
        self.model_metrics[name] = ModelMetrics()

        # Insert into priority list maintaining order
        inserted = False
        for i, existing_name in enumerate(self.model_priority):
            if priority < self._get_model_priority(existing_name):
                self.model_priority.insert(i, name)
                inserted = True
                break

        if not inserted:
            self.model_priority.append(name)

        logger.info(f"Registered model: {name} (priority: {priority})")

    def _get_model_priority(self, name: str) -> int:
        """Get priority for a model (lower = higher priority)."""
        # Default priorities if not explicitly set
        priority_map = {
            "openaudio-s1-mini": 1,
            "coqui-vits": 2,
            "speecht5": 3
        }
        return priority_map.get(name, 999)
    
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

    def select_best_model(self, criteria: ModelSelectionCriteria) -> Optional[str]:
        """Select the best model based on given criteria.

        Args:
            criteria: Selection criteria including text length, quality requirements, etc.

        Returns:
            Name of the best model, or None if no suitable model found
        """
        logger.debug(f"Selecting model for criteria: {criteria}")

        # Get available models that can handle the request
        candidates = []

        for model_name in self.model_priority:
            if not self._is_model_available(model_name):
                continue

            if not self._can_model_handle_request(model_name, criteria):
                continue

            candidates.append(model_name)

        if not candidates:
            logger.warning("No suitable models found for criteria")
            return None

        # Apply strategy-specific selection
        if criteria.strategy == ModelSelectionStrategy.FAST:
            return self._select_fastest_model(candidates, criteria)
        elif criteria.strategy == ModelSelectionStrategy.QUALITY:
            return self._select_highest_quality_model(candidates, criteria)
        elif criteria.strategy == ModelSelectionStrategy.BALANCED:
            return self._select_balanced_model(candidates, criteria)
        else:  # AUTO
            return self._select_auto_model(candidates, criteria)

    def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available and not circuit-broken."""
        if model_name not in self.model_registry:
            return False

        metrics = self.model_metrics.get(model_name)
        if not metrics:
            return True

        # Check circuit breaker
        if metrics.failure_count >= self.circuit_breaker_threshold:
            if metrics.last_failure_time:
                time_since_failure = time.time() - metrics.last_failure_time
                if time_since_failure < self.circuit_breaker_timeout:
                    logger.debug(f"Model {model_name} is circuit-broken")
                    return False
                else:
                    # Reset circuit breaker
                    metrics.failure_count = 0
                    metrics.last_failure_time = None
                    logger.info(f"Circuit breaker reset for model {model_name}")

        return True

    def _can_model_handle_request(self, model_name: str, criteria: ModelSelectionCriteria) -> bool:
        """Check if model can handle the specific request."""
        try:
            model_class = self.model_registry[model_name]

            # Create temporary instance to check capabilities
            temp_model = model_class(model_path=Path("/tmp"), device="cpu")

            # Check text length
            if criteria.text_length > temp_model.get_max_text_length():
                return False

            # Check voice support
            if criteria.voice not in temp_model.get_supported_voices():
                return False

            return True

        except Exception as e:
            logger.warning(f"Could not check capabilities for {model_name}: {e}")
            return False

    def _select_fastest_model(self, candidates: List[str], criteria: ModelSelectionCriteria) -> str:
        """Select the fastest model from candidates."""
        # Prefer models with lower average RTF
        best_model = candidates[0]
        best_rtf = float('inf')

        for model_name in candidates:
            metrics = self.model_metrics.get(model_name)
            if metrics and metrics.average_rtf > 0:
                if metrics.average_rtf < best_rtf:
                    best_rtf = metrics.average_rtf
                    best_model = model_name

        logger.debug(f"Selected fastest model: {best_model} (RTF: {best_rtf:.3f})")
        return best_model

    def _select_highest_quality_model(self, candidates: List[str], criteria: ModelSelectionCriteria) -> str:
        """Select the highest quality model from candidates."""
        # Prefer models with better success rates and known quality
        quality_order = ["openaudio-s1-mini", "coqui-vits", "speecht5"]

        for model_name in quality_order:
            if model_name in candidates:
                logger.debug(f"Selected quality model: {model_name}")
                return model_name

        # Fallback to first candidate
        return candidates[0]

    def _select_balanced_model(self, candidates: List[str], criteria: ModelSelectionCriteria) -> str:
        """Select a balanced model considering both speed and quality."""
        # Score models based on RTF and success rate
        best_model = candidates[0]
        best_score = -1

        for model_name in candidates:
            metrics = self.model_metrics.get(model_name)
            if metrics:
                # Balanced score: success_rate / (1 + rtf)
                rtf = max(metrics.average_rtf, 0.1)  # Avoid division by zero
                score = metrics.success_rate / (1 + rtf)

                if score > best_score:
                    best_score = score
                    best_model = model_name

        logger.debug(f"Selected balanced model: {best_model} (score: {best_score:.3f})")
        return best_model

    def _select_auto_model(self, candidates: List[str], criteria: ModelSelectionCriteria) -> str:
        """Automatically select model based on text characteristics."""
        # Auto selection logic based on text length and requirements

        if criteria.text_length <= 50:
            # Short text: prefer fast models
            return self._select_fastest_model(candidates, criteria)
        elif criteria.text_length <= 200:
            # Medium text: prefer balanced
            return self._select_balanced_model(candidates, criteria)
        else:
            # Long text: prefer quality and stability
            return self._select_highest_quality_model(candidates, criteria)

    def load_model_with_fallback(self, criteria: ModelSelectionCriteria) -> Optional[BaseTTSModel]:
        """Load a model with automatic fallback on failure.

        Args:
            criteria: Selection criteria for model choice

        Returns:
            Loaded model instance or None if all models fail
        """
        selected_model = self.select_best_model(criteria)
        if not selected_model:
            logger.error("No suitable model found for criteria")
            return None

        # Try to load the selected model
        try:
            model = self.load_model(selected_model)
            logger.info(f"Successfully loaded selected model: {selected_model}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load selected model {selected_model}: {e}")
            self._record_model_failure(selected_model)

        # Try fallback models
        for model_name in self.model_priority:
            if model_name == selected_model:
                continue  # Already tried

            if not self._is_model_available(model_name):
                continue

            try:
                model = self.load_model(model_name)
                logger.info(f"Successfully loaded fallback model: {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load fallback model {model_name}: {e}")
                self._record_model_failure(model_name)

        logger.error("All models failed to load")
        return None

    def _record_model_failure(self, model_name: str) -> None:
        """Record a model failure for circuit breaker logic."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics()

        metrics = self.model_metrics[model_name]
        metrics.failure_count += 1
        metrics.last_failure_time = time.time()

        # Update success rate
        metrics.total_requests += 1
        successful_requests = metrics.total_requests - metrics.failure_count
        metrics.success_rate = successful_requests / metrics.total_requests

        logger.warning(f"Recorded failure for {model_name} (failures: {metrics.failure_count})")

    def record_model_success(self, model_name: str, rtf: float) -> None:
        """Record a successful model inference.

        Args:
            model_name: Name of the model
            rtf: Real-time factor achieved
        """
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics()

        metrics = self.model_metrics[model_name]
        metrics.total_requests += 1

        # Update average RTF with exponential moving average
        if metrics.average_rtf == 0:
            metrics.average_rtf = rtf
        else:
            metrics.average_rtf = 0.9 * metrics.average_rtf + 0.1 * rtf

        # Update success rate
        successful_requests = metrics.total_requests - metrics.failure_count
        metrics.success_rate = successful_requests / metrics.total_requests
    
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
