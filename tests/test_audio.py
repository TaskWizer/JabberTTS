"""Tests for JabberTTS audio processing functionality."""

import pytest
import numpy as np
import asyncio

from jabbertts.audio.processor import AudioProcessor, get_audio_processor


class TestAudioProcessor:
    """Test audio processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create an audio processor."""
        return AudioProcessor()
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing."""
        # Generate a simple sine wave
        sample_rate = 16000
        duration = 1.0  # 1 second
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        return audio, sample_rate
    
    def test_processor_creation(self, processor):
        """Test processor creation."""
        assert isinstance(processor, AudioProcessor)
        assert hasattr(processor, 'supported_formats')
        assert len(processor.supported_formats) > 0
    
    def test_processor_singleton(self):
        """Test that get_audio_processor returns the same instance."""
        processor1 = get_audio_processor()
        processor2 = get_audio_processor()
        assert processor1 is processor2
    
    def test_supported_formats(self, processor):
        """Test supported audio formats."""
        expected_formats = ["mp3", "wav", "flac", "opus", "aac", "pcm"]
        
        for fmt in expected_formats:
            assert fmt in processor.supported_formats
    
    def test_processor_info(self, processor):
        """Test getting processor information."""
        info = processor.get_processor_info()
        
        assert isinstance(info, dict)
        assert "supported_formats" in info
        assert "dependencies" in info
        assert "capabilities" in info
        
        # Check dependencies
        deps = info["dependencies"]
        assert "soundfile" in deps
        assert "ffmpeg" in deps
        assert "librosa" in deps
        
        # Check capabilities
        caps = info["capabilities"]
        assert "format_conversion" in caps
        assert "speed_adjustment" in caps
        assert "high_quality_encoding" in caps
    
    def test_normalize_audio(self, processor, sample_audio):
        """Test audio normalization."""
        audio, _ = sample_audio
        
        # Test with different input formats
        test_cases = [
            audio.astype(np.float32),
            audio.astype(np.float64),
            (audio * 32767).astype(np.int16),
        ]
        
        for test_audio in test_cases:
            normalized = processor._normalize_audio(test_audio)
            
            assert normalized.dtype == np.float32
            assert len(normalized.shape) == 1  # Should be mono
            assert np.max(np.abs(normalized)) <= 1.0  # Should be normalized
    
    @pytest.mark.asyncio
    async def test_wav_processing(self, processor, sample_audio):
        """Test WAV format processing."""
        audio, sample_rate = sample_audio
        
        result = await processor.process_audio(
            audio_array=audio,
            sample_rate=sample_rate,
            output_format="wav"
        )
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # WAV files should start with RIFF header
        assert result[:4] == b'RIFF'
        assert result[8:12] == b'WAVE'
    
    @pytest.mark.asyncio
    async def test_pcm_processing(self, processor, sample_audio):
        """Test PCM format processing."""
        audio, sample_rate = sample_audio
        
        result = await processor.process_audio(
            audio_array=audio,
            sample_rate=sample_rate,
            output_format="pcm"
        )
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # PCM should be raw audio data
        # Length should be audio_length * 2 (16-bit samples)
        expected_length = len(audio) * 2
        assert len(result) == expected_length
    
    @pytest.mark.asyncio
    async def test_invalid_format(self, processor, sample_audio):
        """Test processing with invalid format."""
        audio, sample_rate = sample_audio
        
        with pytest.raises(ValueError, match="Unsupported format"):
            await processor.process_audio(
                audio_array=audio,
                sample_rate=sample_rate,
                output_format="invalid_format"
            )
    
    @pytest.mark.asyncio
    async def test_speed_adjustment(self, processor, sample_audio):
        """Test speed adjustment."""
        audio, sample_rate = sample_audio
        
        # Test different speeds
        speeds = [0.5, 1.0, 1.5, 2.0]
        
        for speed in speeds:
            result = await processor.process_audio(
                audio_array=audio,
                sample_rate=sample_rate,
                output_format="wav",
                speed=speed
            )
            
            assert isinstance(result, bytes)
            assert len(result) > 0
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_mp3_processing(self, processor, sample_audio):
        """Test MP3 format processing (slow test)."""
        audio, sample_rate = sample_audio
        
        try:
            result = await processor.process_audio(
                audio_array=audio,
                sample_rate=sample_rate,
                output_format="mp3"
            )
            
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # MP3 files might start with ID3 tag or frame sync
            # Just check that we got some data
            
        except RuntimeError as e:
            # MP3 encoding might fail if ffmpeg is not available
            pytest.skip(f"MP3 encoding failed (ffmpeg not available): {e}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_flac_processing(self, processor, sample_audio):
        """Test FLAC format processing (slow test)."""
        audio, sample_rate = sample_audio
        
        try:
            result = await processor.process_audio(
                audio_array=audio,
                sample_rate=sample_rate,
                output_format="flac"
            )
            
            assert isinstance(result, bytes)
            assert len(result) > 0
            
            # FLAC files should start with fLaC signature
            if processor.has_soundfile:
                assert result[:4] == b'fLaC'
            
        except RuntimeError as e:
            # FLAC encoding might fail if soundfile is not available
            pytest.skip(f"FLAC encoding failed (soundfile not available): {e}")
    
    def test_simple_wav_creation(self, processor, sample_audio):
        """Test simple WAV file creation without dependencies."""
        audio, sample_rate = sample_audio
        
        # Test the fallback WAV creation
        result = processor._create_simple_wav(audio, sample_rate)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Should have WAV header
        assert result[:4] == b'RIFF'
        assert result[8:12] == b'WAVE'
        assert result[12:16] == b'fmt '
        assert result[36:40] == b'data'
    
    def test_audio_normalization_edge_cases(self, processor):
        """Test audio normalization with edge cases."""
        # Test with very loud audio
        loud_audio = np.array([2.0, -3.0, 1.5, -2.5], dtype=np.float32)
        normalized = processor._normalize_audio(loud_audio)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test with very quiet audio
        quiet_audio = np.array([0.01, -0.02, 0.015, -0.005], dtype=np.float32)
        normalized = processor._normalize_audio(quiet_audio)
        assert normalized.dtype == np.float32
        
        # Test with zero audio
        zero_audio = np.zeros(1000, dtype=np.float32)
        normalized = processor._normalize_audio(zero_audio)
        assert np.all(normalized == 0.0)
    
    @pytest.mark.asyncio
    async def test_processing_with_different_sample_rates(self, processor):
        """Test processing with different sample rates."""
        sample_rates = [8000, 16000, 22050, 24000, 44100, 48000]
        
        for sr in sample_rates:
            # Generate test audio
            duration = 0.5  # 0.5 seconds
            t = np.linspace(0, duration, int(sr * duration), False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            result = await processor.process_audio(
                audio_array=audio,
                sample_rate=sr,
                output_format="wav"
            )
            
            assert isinstance(result, bytes)
            assert len(result) > 0


class TestAudioIntegration:
    """Test audio processing integration."""
    
    @pytest.mark.asyncio
    async def test_format_conversion_consistency(self):
        """Test that format conversion produces consistent results."""
        processor = get_audio_processor()
        
        # Generate test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Convert to different formats
        formats_to_test = ["wav", "pcm"]  # Test formats that should always work
        
        results = {}
        for fmt in formats_to_test:
            result = await processor.process_audio(
                audio_array=audio,
                sample_rate=sample_rate,
                output_format=fmt
            )
            results[fmt] = result
            
            assert isinstance(result, bytes)
            assert len(result) > 0
        
        # Results should be different for different formats
        if len(results) > 1:
            format_list = list(results.keys())
            for i in range(len(format_list)):
                for j in range(i + 1, len(format_list)):
                    fmt1, fmt2 = format_list[i], format_list[j]
                    # Different formats should produce different output
                    # (unless they're both raw formats)
                    if fmt1 != "pcm" or fmt2 != "pcm":
                        assert results[fmt1] != results[fmt2]
