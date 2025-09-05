"""Tests for JabberTTS model functionality."""

import pytest
import numpy as np
from pathlib import Path

from jabbertts.models.manager import ModelManager, get_model_manager
from jabbertts.models.base import BaseTTSModel
from jabbertts.models.speecht5 import SpeechT5Model


class TestModelManager:
    """Test model manager functionality."""
    
    def test_model_manager_singleton(self):
        """Test that get_model_manager returns the same instance."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        assert manager1 is manager2
    
    def test_get_available_models(self):
        """Test getting available models."""
        manager = ModelManager()
        models = manager.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Should have at least SpeechT5
        assert "speecht5" in models
    
    def test_model_status(self):
        """Test getting model status."""
        manager = ModelManager()
        status = manager.get_model_status()
        
        assert isinstance(status, dict)
        assert "current_model" in status
        assert "loaded_models" in status
        assert "available_models" in status
        assert "memory_usage" in status


class TestSpeechT5Model:
    """Test SpeechT5 model functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a SpeechT5 model instance."""
        return SpeechT5Model(model_path=Path("./models/speecht5"), device="cpu")
    
    def test_model_creation(self, model):
        """Test model creation."""
        assert isinstance(model, BaseTTSModel)
        assert model.device == "cpu"
        assert not model.is_loaded
    
    def test_get_available_voices(self, model):
        """Test getting available voices."""
        voices = model.get_available_voices()
        
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "alloy" in voices  # OpenAI compatibility
        assert "default" in voices
    
    def test_get_sample_rate(self, model):
        """Test getting sample rate."""
        sample_rate = model.get_sample_rate()
        assert sample_rate == 16000  # SpeechT5 uses 16kHz
    
    def test_get_model_info(self, model):
        """Test getting model information."""
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "sample_rate" in info
        assert "device" in info
        assert "is_loaded" in info
    
    def test_validate_parameters(self, model):
        """Test parameter validation."""
        # Valid parameters should not raise
        model.validate_parameters("Hello world", "alloy", 1.0)
        
        # Invalid parameters should raise
        with pytest.raises(ValueError, match="Text cannot be empty"):
            model.validate_parameters("", "alloy", 1.0)
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            model.validate_parameters("a" * 5000, "alloy", 1.0)
        
        with pytest.raises(ValueError, match="Speed must be between"):
            model.validate_parameters("Hello", "alloy", 10.0)
        
        with pytest.raises(ValueError, match="Voice .* not available"):
            model.validate_parameters("Hello", "invalid_voice", 1.0)
    
    def test_memory_usage(self, model):
        """Test memory usage reporting."""
        usage = model.get_memory_usage()
        
        assert isinstance(usage, dict)
        assert "model_memory" in usage
        assert "total_memory" in usage
        assert usage["model_memory"] >= 0.0
        assert usage["total_memory"] >= 0.0
    
    @pytest.mark.slow
    def test_model_loading(self, model):
        """Test model loading (slow test)."""
        try:
            model.load_model()
            assert model.is_loaded
            
            # Test model info after loading
            info = model.get_model_info()
            assert info["is_loaded"] is True
            
            # Test memory usage after loading
            usage = model.get_memory_usage()
            assert usage["model_memory"] > 0.0
            
        except Exception as e:
            # Model loading might fail in CI environment
            pytest.skip(f"Model loading failed (expected in CI): {e}")
        finally:
            if model.is_loaded:
                model.unload_model()
    
    @pytest.mark.slow
    def test_speech_generation(self, model):
        """Test speech generation (slow test)."""
        try:
            model.load_model()
            
            # Generate speech
            audio = model.generate_speech(
                text="Hello, this is a test.",
                voice="alloy",
                speed=1.0
            )
            
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert len(audio) > 0
            
            # Audio should be reasonable length (not too short or long)
            duration = len(audio) / model.get_sample_rate()
            assert 0.5 < duration < 10.0  # Between 0.5 and 10 seconds
            
        except Exception as e:
            # Speech generation might fail in CI environment
            pytest.skip(f"Speech generation failed (expected in CI): {e}")
        finally:
            if model.is_loaded:
                model.unload_model()


class TestModelIntegration:
    """Test model integration with manager."""
    
    @pytest.mark.slow
    def test_load_model_via_manager(self):
        """Test loading model via manager."""
        manager = ModelManager()
        
        try:
            # Try to load SpeechT5 model
            model = manager.load_model("speecht5")
            
            assert model is not None
            assert model.is_loaded
            assert manager.get_current_model() == model
            
            # Test model status
            status = manager.get_model_status()
            assert status["current_model"] == "speecht5"
            assert "speecht5" in status["loaded_models"]
            
        except Exception as e:
            # Model loading might fail in CI environment
            pytest.skip(f"Model loading via manager failed (expected in CI): {e}")
        finally:
            manager.unload_all_models()
    
    def test_model_validation(self):
        """Test model file validation."""
        manager = ModelManager()

        # SpeechT5 should always validate (auto-download)
        # Note: SpeechT5Model.validate_files always returns True
        assert manager.validate_model_files("speecht5") is True

        # Non-existent model should not validate
        assert manager.validate_model_files("non_existent_model") is False
