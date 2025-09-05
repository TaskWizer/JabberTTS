"""Audio Processing for JabberTTS.

This module handles audio format conversion, encoding, and processing
for the generated TTS audio output.
"""

import io
import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processor for TTS output.
    
    Handles format conversion, encoding, and audio processing operations
    for the generated speech audio.
    """
    
    def __init__(self):
        """Initialize audio processor."""
        self.supported_formats = ["mp3", "wav", "flac", "opus", "aac", "pcm"]
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check for required audio processing dependencies."""
        self.has_soundfile = False
        self.has_ffmpeg = False
        self.has_librosa = False
        
        try:
            import soundfile
            self.has_soundfile = True
            logger.info("soundfile available for audio processing")
        except ImportError:
            logger.warning("soundfile not available, limited audio format support")
        
        try:
            import ffmpeg
            self.has_ffmpeg = True
            logger.info("ffmpeg-python available for audio encoding")
        except ImportError:
            logger.warning("ffmpeg-python not available, limited format conversion")
        
        try:
            import librosa
            self.has_librosa = True
            logger.info("librosa available for audio processing")
        except ImportError:
            logger.warning("librosa not available, limited audio effects")
    
    async def process_audio(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        output_format: str = "mp3",
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """Process audio array to the requested format.
        
        Args:
            audio_array: Input audio as numpy array (float32)
            sample_rate: Sample rate of the input audio
            output_format: Desired output format
            speed: Speed adjustment (already applied in TTS, but can be re-applied)
            **kwargs: Additional processing parameters
            
        Returns:
            Processed audio data as bytes
            
        Raises:
            ValueError: If format is not supported
            RuntimeError: If processing fails
        """
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}")
        
        try:
            logger.debug(f"Processing audio: {audio_array.shape} samples at {sample_rate}Hz to {output_format}")
            
            # Ensure audio is in the correct format
            audio_array = self._normalize_audio(audio_array)
            
            # Apply speed adjustment if needed (and different from 1.0)
            if speed != 1.0 and self.has_librosa:
                audio_array = self._adjust_speed(audio_array, speed)
            
            # Convert to the requested format
            if output_format == "wav":
                return self._to_wav(audio_array, sample_rate)
            elif output_format == "mp3":
                return self._to_mp3(audio_array, sample_rate)
            elif output_format == "flac":
                return self._to_flac(audio_array, sample_rate)
            elif output_format == "opus":
                return self._to_opus(audio_array, sample_rate)
            elif output_format == "aac":
                return self._to_aac(audio_array, sample_rate)
            elif output_format == "pcm":
                return self._to_pcm(audio_array)
            else:
                raise ValueError(f"Format {output_format} not implemented")
                
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise RuntimeError(f"Audio processing failed: {e}") from e
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio array.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        # Ensure float32 format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure mono (if stereo, take first channel)
        if len(audio.shape) > 1:
            audio = audio[:, 0] if audio.shape[1] > 0 else audio.flatten()
        
        # Normalize to [-1, 1] range
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Adjust audio speed using librosa.
        
        Args:
            audio: Input audio array
            speed: Speed multiplier
            
        Returns:
            Speed-adjusted audio
        """
        if not self.has_librosa:
            logger.warning("librosa not available, speed adjustment skipped")
            return audio
        
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except Exception as e:
            logger.warning(f"Speed adjustment failed: {e}")
            return audio
    
    def _to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio to WAV format.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            WAV audio data
        """
        if self.has_soundfile:
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV', subtype='PCM_16')
            return buffer.getvalue()
        else:
            # Fallback: simple WAV header + PCM data
            return self._create_simple_wav(audio, sample_rate)
    
    def _to_mp3(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio to MP3 format.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            MP3 audio data
        """
        if self.has_ffmpeg:
            return self._ffmpeg_encode(audio, sample_rate, "mp3", {"audio_bitrate": "96k"})
        else:
            logger.warning("ffmpeg not available, returning WAV instead of MP3")
            return self._to_wav(audio, sample_rate)
    
    def _to_flac(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio to FLAC format.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            FLAC audio data
        """
        if self.has_soundfile:
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='FLAC')
            return buffer.getvalue()
        else:
            logger.warning("soundfile not available, returning WAV instead of FLAC")
            return self._to_wav(audio, sample_rate)
    
    def _to_opus(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio to Opus format.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Opus audio data
        """
        if self.has_ffmpeg:
            return self._ffmpeg_encode(audio, sample_rate, "opus", {"audio_bitrate": "64k"})
        else:
            logger.warning("ffmpeg not available, returning WAV instead of Opus")
            return self._to_wav(audio, sample_rate)
    
    def _to_aac(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio to AAC format.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            AAC audio data
        """
        if self.has_ffmpeg:
            return self._ffmpeg_encode(audio, sample_rate, "aac", {"audio_bitrate": "128k"})
        else:
            logger.warning("ffmpeg not available, returning WAV instead of AAC")
            return self._to_wav(audio, sample_rate)
    
    def _to_pcm(self, audio: np.ndarray) -> bytes:
        """Convert audio to raw PCM format.
        
        Args:
            audio: Audio array
            
        Returns:
            PCM audio data
        """
        # Convert to 16-bit PCM
        pcm_data = (audio * 32767).astype(np.int16)
        return pcm_data.tobytes()
    
    def _ffmpeg_encode(self, audio: np.ndarray, sample_rate: int, format: str, options: Dict[str, str]) -> bytes:
        """Encode audio using ffmpeg.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            format: Output format
            options: Encoding options
            
        Returns:
            Encoded audio data
        """
        try:
            import ffmpeg
            
            # Create input stream from numpy array
            input_stream = ffmpeg.input('pipe:', format='f32le', acodec='pcm_f32le', ar=sample_rate, ac=1)
            
            # Create output stream with specified format and options
            output_stream = ffmpeg.output(input_stream, 'pipe:', format=format, **options)
            
            # Run ffmpeg
            stdout, stderr = ffmpeg.run(output_stream, input=audio.tobytes(), capture_stdout=True, capture_stderr=True)

            if stderr:
                logger.warning(f"FFmpeg stderr: {stderr.decode()}")

            return stdout
            
        except Exception as e:
            logger.error(f"FFmpeg encoding failed: {e}")
            raise RuntimeError(f"Audio encoding failed: {e}") from e
    
    def _create_simple_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Create a simple WAV file without external dependencies.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            WAV file data
        """
        # Convert to 16-bit PCM
        pcm_data = (audio * 32767).astype(np.int16)
        
        # WAV header
        header = bytearray()
        header.extend(b'RIFF')
        header.extend((36 + len(pcm_data) * 2).to_bytes(4, 'little'))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))  # PCM format chunk size
        header.extend((1).to_bytes(2, 'little'))   # PCM format
        header.extend((1).to_bytes(2, 'little'))   # Mono
        header.extend(sample_rate.to_bytes(4, 'little'))
        header.extend((sample_rate * 2).to_bytes(4, 'little'))  # Byte rate
        header.extend((2).to_bytes(2, 'little'))   # Block align
        header.extend((16).to_bytes(2, 'little'))  # Bits per sample
        header.extend(b'data')
        header.extend((len(pcm_data) * 2).to_bytes(4, 'little'))
        
        return bytes(header) + pcm_data.tobytes()
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get audio processor information.
        
        Returns:
            Dictionary with processor capabilities
        """
        return {
            "supported_formats": self.supported_formats,
            "dependencies": {
                "soundfile": self.has_soundfile,
                "ffmpeg": self.has_ffmpeg,
                "librosa": self.has_librosa
            },
            "capabilities": {
                "format_conversion": True,
                "speed_adjustment": self.has_librosa,
                "high_quality_encoding": self.has_ffmpeg
            }
        }


# Global audio processor instance
_audio_processor: Optional[AudioProcessor] = None


def get_audio_processor() -> AudioProcessor:
    """Get the global audio processor instance.
    
    Returns:
        Global AudioProcessor instance
    """
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
