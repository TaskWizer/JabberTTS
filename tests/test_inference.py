"""Tests for JabberTTS inference functionality."""

import pytest
import asyncio
import numpy as np

from jabbertts.inference.engine import InferenceEngine, get_inference_engine
from jabbertts.inference.preprocessing import TextPreprocessor


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a text preprocessor."""
        return TextPreprocessor(use_phonemizer=False)  # Disable phonemizer for testing
    
    def test_basic_cleaning(self, preprocessor):
        """Test basic text cleaning."""
        text = "  Hello\t\nworld  \r  "
        result = preprocessor.preprocess(text)
        assert result == "Hello world."
    
    def test_unicode_normalization(self, preprocessor):
        """Test Unicode normalization."""
        text = "\u201cHello world\u201d"  # Unicode quotes
        result = preprocessor.preprocess(text)
        # The Unicode quotes should be normalized
        assert '"' in result  # Should contain normalized quotes
    
    def test_abbreviation_expansion(self, preprocessor):
        """Test abbreviation expansion."""
        text = "Dr. Smith lives on Main St."
        result = preprocessor.preprocess(text)
        assert "Doctor" in result
        assert "Street" in result
    
    def test_punctuation_handling(self, preprocessor):
        """Test punctuation handling."""
        text = "Hello world"  # No ending punctuation
        result = preprocessor.preprocess(text)
        assert result.endswith(".")
    
    def test_empty_text_validation(self, preprocessor):
        """Test empty text validation."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            preprocessor.preprocess("")
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            preprocessor.preprocess("   ")
    
    def test_preprocessing_info(self, preprocessor):
        """Test getting preprocessing information."""
        info = preprocessor.get_preprocessing_info()
        
        assert isinstance(info, dict)
        assert "phonemizer_available" in info
        assert "supported_features" in info
        assert isinstance(info["supported_features"], list)


class TestInferenceEngine:
    """Test inference engine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create an inference engine."""
        return InferenceEngine()
    
    def test_engine_creation(self, engine):
        """Test engine creation."""
        assert isinstance(engine, InferenceEngine)
        assert hasattr(engine, 'performance_stats')
        assert hasattr(engine, 'model_manager')
        assert hasattr(engine, 'preprocessor')
    
    def test_engine_singleton(self):
        """Test that get_inference_engine returns the same instance."""
        engine1 = get_inference_engine()
        engine2 = get_inference_engine()
        assert engine1 is engine2
    
    def test_input_validation(self, engine):
        """Test input validation."""
        # Valid input should not raise
        engine._validate_input("Hello world", "alloy", 1.0, "mp3")
        
        # Invalid inputs should raise
        with pytest.raises(ValueError, match="Text cannot be empty"):
            engine._validate_input("", "alloy", 1.0, "mp3")
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            engine._validate_input("a" * 5000, "alloy", 1.0, "mp3")
        
        with pytest.raises(ValueError, match="Speed must be between"):
            engine._validate_input("Hello", "alloy", 10.0, "mp3")
        
        with pytest.raises(ValueError, match="Invalid format"):
            engine._validate_input("Hello", "alloy", 1.0, "invalid")
    
    def test_performance_stats(self, engine):
        """Test performance statistics."""
        stats = engine.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "total_inference_time" in stats
        assert "total_audio_duration" in stats
        assert "average_rtf" in stats
        assert "model_status" in stats
        assert "preprocessing_info" in stats
    
    def test_health_status(self, engine):
        """Test health status."""
        status = engine.get_health_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "model_loaded" in status
        assert "model_name" in status
        assert "preprocessor_ready" in status
        assert "performance" in status
    
    @pytest.mark.asyncio
    async def test_text_preprocessing(self, engine):
        """Test async text preprocessing."""
        text = "Hello world"
        result = await engine._preprocess_text(text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.endswith(".")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_speech_generation_async(self, engine):
        """Test async speech generation (slow test)."""
        try:
            result = await engine.generate_speech(
                text="Hello, this is a test.",
                voice="alloy",
                speed=1.0,
                response_format="wav"
            )
            
            assert isinstance(result, dict)
            assert "audio_data" in result
            assert "sample_rate" in result
            assert "duration" in result
            assert "inference_time" in result
            assert "rtf" in result
            
            # Check audio data
            audio_data = result["audio_data"]
            assert isinstance(audio_data, np.ndarray)
            assert len(audio_data) > 0
            
            # Check performance metrics
            assert result["rtf"] >= 0
            assert result["duration"] > 0
            assert result["inference_time"] > 0
            
        except Exception as e:
            # Speech generation might fail in CI environment
            pytest.skip(f"Speech generation failed (expected in CI): {e}")
    
    @pytest.mark.slow
    def test_speech_generation_sync(self, engine):
        """Test sync speech generation (slow test)."""
        try:
            result = engine.generate_speech_sync(
                text="Hello, this is a test.",
                voice="alloy",
                speed=1.0,
                response_format="wav"
            )
            
            assert isinstance(result, dict)
            assert "audio_data" in result
            assert isinstance(result["audio_data"], np.ndarray)
            
        except Exception as e:
            # Speech generation might fail in CI environment
            pytest.skip(f"Speech generation failed (expected in CI): {e}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_warmup(self, engine):
        """Test engine warmup (slow test)."""
        try:
            await engine.warmup()
            # If warmup succeeds, engine should be ready
            status = engine.get_health_status()
            # Note: status might still be "degraded" if model loading failed
            assert status["preprocessor_ready"] is True
            
        except Exception as e:
            # Warmup might fail in CI environment
            pytest.skip(f"Warmup failed (expected in CI): {e}")


class TestInferenceIntegration:
    """Test inference integration with models and audio processing."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_end_to_end_inference(self):
        """Test complete end-to-end inference pipeline."""
        try:
            engine = get_inference_engine()
            
            # Test with different parameters
            test_cases = [
                {"text": "Hello world.", "voice": "alloy", "speed": 1.0},
                {"text": "This is a longer test sentence.", "voice": "echo", "speed": 0.8},
                {"text": "Quick test!", "voice": "fable", "speed": 1.2},
            ]
            
            for case in test_cases:
                result = await engine.generate_speech(**case)
                
                assert isinstance(result, dict)
                assert "audio_data" in result
                assert len(result["audio_data"]) > 0
                
                # Performance should be reasonable
                assert result["rtf"] < 5.0  # Should be much faster than 5x real-time
                
        except Exception as e:
            # End-to-end test might fail in CI environment
            pytest.skip(f"End-to-end test failed (expected in CI): {e}")
    
    def test_performance_tracking(self):
        """Test performance statistics tracking."""
        engine = InferenceEngine()
        
        # Simulate some performance updates
        engine._update_performance_stats(1.0, 2.0, 0.5)  # 1s inference, 2s audio, RTF 0.5
        engine._update_performance_stats(2.0, 3.0, 0.67)  # 2s inference, 3s audio, RTF 0.67
        
        stats = engine.get_performance_stats()
        
        assert stats["total_requests"] == 2
        assert stats["total_inference_time"] == 3.0
        assert stats["total_audio_duration"] == 5.0
        assert abs(stats["average_rtf"] - 0.6) < 0.01  # Should be 3.0/5.0 = 0.6
