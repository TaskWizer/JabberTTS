"""Audio Processing for JabberTTS.

This module handles audio format conversion, encoding, and processing
for the generated TTS audio output.
"""

import io
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
from jabbertts.config import get_settings

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processor for TTS output.
    
    Handles format conversion, encoding, and audio processing operations
    for the generated speech audio.
    """
    
    def __init__(self):
        """Initialize audio processor."""
        self.supported_formats = ["mp3", "wav", "flac", "opus", "aac", "pcm"]
        self.settings = get_settings()
        self._check_dependencies()
        self._setup_quality_presets()
    
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

    def _setup_quality_presets(self) -> None:
        """Setup audio quality presets based on configuration.

        CRITICAL FIX: Use model native sample rate (16kHz) to prevent quality degradation
        from aggressive upsampling. Quality differences come from bitrates and processing,
        not sample rate upsampling which introduces artifacts.
        """
        self.quality_presets = {
            "low": {
                "sample_rate": 16000,  # Match SpeechT5 native rate
                "bitrates": {"mp3": "64k", "aac": "48k", "opus": "32k"},
                "enable_enhancement": False,
                "noise_reduction": False,
                "compression": False
            },
            "standard": {
                "sample_rate": 16000,  # FIXED: Use native rate, not 24kHz
                "bitrates": {"mp3": "128k", "aac": "96k", "opus": "64k"},  # Higher bitrates for quality
                "enable_enhancement": True,
                "noise_reduction": False,  # FIXED: Disable aggressive noise reduction
                "compression": False       # FIXED: Disable compression that causes artifacts
            },
            "high": {
                "sample_rate": 16000,  # FIXED: Use native rate, not 44.1kHz
                "bitrates": {"mp3": "192k", "aac": "128k", "opus": "96k"},
                "enable_enhancement": True,
                "noise_reduction": False,  # FIXED: Disable to prevent artifacts
                "compression": False       # FIXED: Disable to prevent artifacts
            },
            "ultra": {
                "sample_rate": 16000,  # FIXED: Use native rate, not 48kHz
                "bitrates": {"mp3": "320k", "aac": "256k", "opus": "128k"},
                "enable_enhancement": True,
                "noise_reduction": False,  # FIXED: Disable to prevent artifacts
                "compression": False       # FIXED: Disable to prevent artifacts
            }
        }

        # Get current quality preset
        self.current_preset = self.quality_presets.get(
            self.settings.audio_quality,
            self.quality_presets["standard"]
        )
    
    async def process_audio(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        output_format: str = "mp3",
        speed: float = 1.0,
        **kwargs
    ) -> tuple[bytes, dict]:
        """Process audio array to the requested format.

        Args:
            audio_array: Input audio as numpy array (float32)
            sample_rate: Sample rate of the input audio
            output_format: Desired output format
            speed: Speed adjustment (already applied in TTS, but can be re-applied)
            **kwargs: Additional processing parameters

        Returns:
            Tuple of (processed audio data as bytes, metadata dict with duration info)

        Raises:
            ValueError: If format is not supported
            RuntimeError: If processing fails
        """
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}")
        
        try:
            logger.debug(f"Processing audio: {audio_array.shape} samples at {sample_rate}Hz to {output_format}")

            # Store original duration for comparison
            original_duration = len(audio_array) / sample_rate

            # Ensure audio is in the correct format
            audio_array = self._normalize_audio(audio_array)

            # Apply audio enhancements if enabled
            if self.settings.enable_audio_enhancement and self.current_preset["enable_enhancement"]:
                audio_array = self._enhance_audio(audio_array, sample_rate)

            # Apply speed adjustment if needed (and different from 1.0)
            if speed != 1.0 and self.has_librosa:
                audio_array = self._adjust_speed(audio_array, speed)

            # Resample if needed for quality preset
            target_sample_rate = self._get_target_sample_rate(output_format, sample_rate)
            if target_sample_rate != sample_rate:
                audio_array, sample_rate = self._resample_audio(audio_array, sample_rate, target_sample_rate)
            
            # Calculate final processed duration
            processed_duration = len(audio_array) / sample_rate

            # Create metadata
            metadata = {
                "original_duration": original_duration,
                "processed_duration": processed_duration,
                "original_sample_rate": kwargs.get("original_sample_rate", sample_rate),
                "final_sample_rate": sample_rate,
                "speed_applied": speed,
                "enhancement_applied": self.settings.enable_audio_enhancement and self.current_preset["enable_enhancement"]
            }

            # Convert to the requested format
            if output_format == "wav":
                audio_data = self._to_wav(audio_array, sample_rate)
            elif output_format == "mp3":
                audio_data = self._to_mp3(audio_array, sample_rate)
            elif output_format == "flac":
                audio_data = self._to_flac(audio_array, sample_rate)
            elif output_format == "opus":
                audio_data = self._to_opus(audio_array, sample_rate)
            elif output_format == "aac":
                audio_data = self._to_aac(audio_array, sample_rate)
            elif output_format == "pcm":
                audio_data = self._to_pcm(audio_array)
            else:
                raise ValueError(f"Format {output_format} not implemented")

            return audio_data, metadata
                
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

    def _enhance_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply audio enhancements to improve quality.

        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio

        Returns:
            Enhanced audio array
        """
        enhanced_audio = audio.copy()

        try:
            # Apply noise reduction if enabled
            if self.settings.noise_reduction and self.current_preset["noise_reduction"]:
                enhanced_audio = self._reduce_noise(enhanced_audio, sample_rate)

            # Apply dynamic range compression if enabled
            if self.settings.dynamic_range_compression and self.current_preset["compression"]:
                enhanced_audio = self._apply_compression(enhanced_audio)

            # Apply advanced normalization
            enhanced_audio = self._advanced_normalize(enhanced_audio)

            # CRITICAL FIX: Disable stereo enhancement for TTS audio
            # Stereo enhancement corrupts mono TTS audio and causes dimension mismatches
            # TTS audio should remain mono for optimal quality and compatibility
            # if self.settings.stereo_enhancement:
            #     enhanced_audio = self._enhance_stereo(enhanced_audio)

            logger.debug("Audio enhancement applied successfully")
            return enhanced_audio

        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}, using original audio")
            return audio

    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction to audio.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Noise-reduced audio
        """
        if not self.has_librosa:
            return audio

        try:
            import librosa

            # Simple spectral gating noise reduction
            # Estimate noise floor from the first 0.5 seconds
            noise_sample_length = min(int(0.5 * sample_rate), len(audio) // 4)
            noise_floor = np.mean(np.abs(audio[:noise_sample_length]))

            # Apply gentle high-pass filter to remove low-frequency noise
            if sample_rate >= 16000:
                audio = librosa.effects.preemphasis(audio, coef=0.97)

            # Spectral subtraction (simplified)
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Reduce magnitude where it's close to noise floor
            noise_threshold = noise_floor * 2.0
            magnitude = np.where(magnitude < noise_threshold,
                               magnitude * 0.5, magnitude)

            # Reconstruct audio
            enhanced_stft = magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)

            return enhanced_audio

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio

    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression.

        Args:
            audio: Input audio array

        Returns:
            Compressed audio
        """
        try:
            # Simple soft compression
            threshold = 0.7
            ratio = 4.0

            # Find peaks above threshold
            abs_audio = np.abs(audio)
            above_threshold = abs_audio > threshold

            if np.any(above_threshold):
                # Apply compression to peaks
                compressed_magnitude = threshold + (abs_audio - threshold) / ratio
                compressed_audio = np.where(
                    above_threshold,
                    np.sign(audio) * compressed_magnitude,
                    audio
                )
                return compressed_audio

            return audio

        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return audio

    def _advanced_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced audio normalization.

        CRITICAL FIX: Conservative normalization to prevent clipping and artifacts.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio (guaranteed <1.0 peak to prevent clipping)
        """
        try:
            if self.settings.audio_normalization == "peak":
                # FIXED: More conservative peak normalization
                max_val = np.abs(audio).max()
                if max_val > 0:
                    # Use 0.85 instead of 0.95 for more headroom to prevent clipping
                    normalized = audio / max_val * 0.85
                    logger.debug(f"Peak normalization: {max_val:.3f} -> {np.abs(normalized).max():.3f}")
                    return normalized

            elif self.settings.audio_normalization == "rms":
                # FIXED: More conservative RMS normalization
                rms = np.sqrt(np.mean(audio**2))
                if rms > 0:
                    target_rms = 0.15  # FIXED: Lower target to prevent clipping
                    normalized = audio * (target_rms / rms)
                    # Ensure no clipping
                    max_val = np.abs(normalized).max()
                    if max_val > 0.85:
                        normalized = normalized / max_val * 0.85
                    logger.debug(f"RMS normalization: {rms:.3f} -> {np.sqrt(np.mean(normalized**2)):.3f}")
                    return normalized

            elif self.settings.audio_normalization == "lufs":
                # FIXED: More conservative LUFS-like normalization
                if self.has_librosa:
                    import librosa
                    # Apply A-weighting-like filter
                    audio_filtered = librosa.effects.preemphasis(audio)
                    rms = np.sqrt(np.mean(audio_filtered**2))
                    if rms > 0:
                        target_lufs = 0.15  # FIXED: Lower target to prevent clipping
                        normalized = audio * (target_lufs / rms)
                        # Ensure no clipping
                        max_val = np.abs(normalized).max()
                        if max_val > 0.85:
                            normalized = normalized / max_val * 0.85
                        logger.debug(f"LUFS normalization: {rms:.3f} -> peak {np.abs(normalized).max():.3f}")
                        return normalized

            # Fallback: basic normalization with conservative headroom
            max_val = np.abs(audio).max()
            if max_val > 0.85:
                return audio / max_val * 0.85
            return audio

        except Exception as e:
            logger.warning(f"Advanced normalization failed: {e}")
            return self._normalize_audio(audio)  # Fallback to basic normalization

    def _enhance_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Convert mono audio to enhanced stereo.

        Args:
            audio: Input mono audio array

        Returns:
            Stereo audio array (2D: [samples, 2])
        """
        try:
            if len(audio.shape) > 1:
                return audio  # Already stereo

            # Create stereo effect with slight delay and filtering
            left_channel = audio

            # Create right channel with slight delay and high-frequency emphasis
            delay_samples = int(0.001 * 24000)  # 1ms delay
            right_channel = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]

            # Apply subtle high-frequency boost to right channel
            if self.has_librosa:
                import librosa
                right_channel = librosa.effects.preemphasis(right_channel, coef=0.95)

            # Combine channels
            stereo_audio = np.column_stack([left_channel, right_channel])
            return stereo_audio

        except Exception as e:
            logger.warning(f"Stereo enhancement failed: {e}")
            return audio

    def _get_target_sample_rate(self, output_format: str, current_rate: int) -> int:
        """Get target sample rate based on quality preset and format.

        CRITICAL FIX: Always preserve original sample rate to prevent quality degradation.
        Upsampling TTS audio introduces artifacts and does not improve quality.

        Args:
            output_format: Output audio format
            current_rate: Current sample rate

        Returns:
            Target sample rate (always original rate for TTS audio)
        """
        # CRITICAL FIX: Always use original sample rate for TTS audio
        # Resampling TTS audio (especially upsampling) introduces artifacts
        # and does not improve quality. The model's native rate is optimal.

        logger.debug(f"Preserving original sample rate {current_rate}Hz for format {output_format}")
        return current_rate

        # OLD LOGIC DISABLED - was causing quality degradation:
        # preset_rate = self.current_preset["sample_rate"]
        # if output_format in ["flac", "wav"]:
        #     return current_rate
        # if 8000 <= current_rate <= 48000:
        #     if abs(current_rate - preset_rate) / preset_rate < 0.5:
        #         return current_rate
        # return preset_rate

    def _resample_audio(self, audio: np.ndarray, current_rate: int, target_rate: int) -> Tuple[np.ndarray, int]:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio array
            current_rate: Current sample rate
            target_rate: Target sample rate

        Returns:
            Tuple of (resampled_audio, target_rate)
        """
        if current_rate == target_rate:
            return audio, current_rate

        if not self.has_librosa:
            logger.warning("librosa not available, skipping resampling")
            return audio, current_rate

        try:
            import librosa

            # Handle stereo audio
            if len(audio.shape) > 1:
                resampled_channels = []
                for channel in range(audio.shape[1]):
                    resampled_channel = librosa.resample(
                        audio[:, channel],
                        orig_sr=current_rate,
                        target_sr=target_rate,
                        res_type='kaiser_best'
                    )
                    resampled_channels.append(resampled_channel)
                resampled_audio = np.column_stack(resampled_channels)
            else:
                resampled_audio = librosa.resample(
                    audio,
                    orig_sr=current_rate,
                    target_sr=target_rate,
                    res_type='kaiser_best'
                )

            logger.debug(f"Resampled audio from {current_rate}Hz to {target_rate}Hz")
            return resampled_audio, target_rate

        except Exception as e:
            logger.warning(f"Resampling failed: {e}")
            return audio, current_rate
    
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
            bitrate = self._get_bitrate("mp3", audio)
            return self._ffmpeg_encode(audio, sample_rate, "mp3", {"audio_bitrate": bitrate})
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
            bitrate = self._get_bitrate("opus", audio)
            return self._ffmpeg_encode(audio, sample_rate, "opus", {"audio_bitrate": bitrate})
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
            bitrate = self._get_bitrate("aac", audio)
            return self._ffmpeg_encode(audio, sample_rate, "aac", {"audio_bitrate": bitrate})
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
        """Encode audio using ffmpeg with optimized settings.

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

            # Get format-specific optimization profile
            optimized_options = self._get_format_optimization_profile(format, sample_rate, options)

            # Create input stream from numpy array
            input_stream = ffmpeg.input(
                'pipe:',
                format='f32le',
                acodec='pcm_f32le',
                ar=sample_rate,
                ac=1
            )

            # Create output stream with optimized options
            output_stream = ffmpeg.output(input_stream, 'pipe:', format=format, **optimized_options)

            # Run ffmpeg with optimized settings
            stdout, stderr = ffmpeg.run(
                output_stream,
                input=audio.tobytes(),
                capture_stdout=True,
                capture_stderr=True,
                quiet=True  # Reduce noise unless there's an error
            )

            if stderr:
                stderr_text = stderr.decode()
                # Only log actual errors, not informational messages
                if any(level in stderr_text.lower() for level in ['error', 'fatal']):
                    logger.warning(f"FFmpeg error: {stderr_text}")
                else:
                    logger.debug(f"FFmpeg info: {stderr_text}")

            return stdout

        except Exception as e:
            logger.error(f"FFmpeg encoding failed for format {format}: {e}")
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
            },
            "quality_presets": self.quality_presets,
            "current_preset": self.current_preset,
            "audio_enhancement": {
                "enabled": self.settings.enable_audio_enhancement,
                "noise_reduction": self.settings.noise_reduction,
                "compression": self.settings.dynamic_range_compression,
                "normalization": self.settings.audio_normalization,
                "stereo_enhancement": self.settings.stereo_enhancement
            }
        }

    def _get_bitrate(self, format: str, audio: Optional[np.ndarray] = None) -> str:
        """Get bitrate for a format based on quality settings and content analysis.

        Args:
            format: Audio format
            audio: Optional audio array for content-based adaptation

        Returns:
            Bitrate string (e.g., "128k")
        """
        if self.settings.bitrate_quality == "adaptive":
            # Use adaptive bitrate based on content and preset
            base_bitrate = self.current_preset["bitrates"].get(format, "96k")

            if audio is not None:
                # Analyze audio content for adaptive bitrate selection
                adapted_bitrate = self._analyze_content_for_bitrate(audio, format, base_bitrate)
                return adapted_bitrate

            return base_bitrate

        # Manual bitrate settings
        bitrate_map = {
            "low": {"mp3": "64k", "aac": "48k", "opus": "32k"},
            "medium": {"mp3": "128k", "aac": "96k", "opus": "64k"},
            "high": {"mp3": "256k", "aac": "192k", "opus": "128k"}
        }

        return bitrate_map.get(self.settings.bitrate_quality, {}).get(format, "96k")

    def _analyze_content_for_bitrate(self, audio: np.ndarray, format: str, base_bitrate: str) -> str:
        """Analyze audio content to determine optimal bitrate.

        Args:
            audio: Audio array
            format: Output format
            base_bitrate: Base bitrate from preset

        Returns:
            Optimized bitrate string
        """
        try:
            # Calculate audio characteristics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            dynamic_range = 20 * np.log10(peak / (rms + 1e-8))

            # Calculate spectral characteristics if possible
            spectral_complexity = 1.0  # Default complexity

            try:
                # Simple spectral analysis using FFT
                fft = np.fft.rfft(audio)
                magnitude = np.abs(fft)

                # Calculate spectral centroid (brightness)
                freqs = np.fft.rfftfreq(len(audio))
                spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)

                # Calculate spectral spread (complexity)
                spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8))

                # Normalize complexity measure (0.5 to 1.5 range)
                spectral_complexity = 0.5 + min(1.0, spectral_spread * 1000)

            except Exception:
                pass  # Use default complexity if analysis fails

            # Extract base bitrate number
            base_num = int(base_bitrate.rstrip('k'))

            # Adaptation factors
            dynamic_factor = min(1.2, max(0.8, dynamic_range / 20))  # 0.8-1.2 based on dynamic range
            complexity_factor = min(1.3, max(0.7, spectral_complexity))  # 0.7-1.3 based on complexity

            # Format-specific adjustments
            format_factors = {
                "mp3": 1.0,    # MP3 baseline
                "aac": 0.85,   # AAC is more efficient
                "opus": 0.7,   # Opus is very efficient for speech
                "flac": 1.0    # Lossless, no bitrate adaptation
            }

            format_factor = format_factors.get(format, 1.0)

            # Calculate adapted bitrate
            if format != "flac":  # Don't adapt lossless formats
                adapted_num = int(base_num * dynamic_factor * complexity_factor * format_factor)

                # Clamp to reasonable ranges per format
                format_ranges = {
                    "mp3": (64, 320),
                    "aac": (48, 256),
                    "opus": (32, 128)
                }

                min_br, max_br = format_ranges.get(format, (64, 256))
                adapted_num = max(min_br, min(max_br, adapted_num))

                logger.debug(f"Adaptive bitrate for {format}: {base_bitrate} -> {adapted_num}k "
                           f"(dynamic: {dynamic_factor:.2f}, complexity: {complexity_factor:.2f})")

                return f"{adapted_num}k"

            return base_bitrate

        except Exception as e:
            logger.warning(f"Content analysis failed, using base bitrate: {e}")
            return base_bitrate

    def _get_format_optimization_profile(self, format: str, sample_rate: int, base_options: Dict[str, str]) -> Dict[str, str]:
        """Get optimized encoding options for specific format.

        Args:
            format: Audio format (mp3, aac, opus, etc.)
            sample_rate: Sample rate
            base_options: Base encoding options

        Returns:
            Optimized encoding options
        """
        # Start with base options
        options = base_options.copy()

        # Format-specific optimizations (simplified for compatibility)
        if format == "mp3":
            # MP3 optimization for speech
            options.update({
                "acodec": "libmp3lame",
                "q:a": "2",  # High quality VBR
            })
            # Use consistent bitrate for speech
            if "audio_bitrate" in options:
                # Keep the bitrate setting as-is for compatibility
                pass

        elif format == "aac":
            # AAC optimization for speech (simplified for compatibility)
            options.update({
                "acodec": "aac",
            })
            # Use standard AAC encoding without profile specification
            if "audio_bitrate" in options:
                # Keep standard bitrate encoding
                pass

        elif format == "opus":
            # Opus optimization for speech
            options.update({
                "acodec": "libopus",
                "application": "voip",  # Optimize for speech
            })
            # Opus handles VBR internally
            if "audio_bitrate" in options:
                options["compression_level"] = "10"  # Maximum compression efficiency

        elif format == "flac":
            # FLAC optimization for lossless compression
            options.update({
                "acodec": "flac",
                "compression_level": "8",  # Maximum compression
            })
            # Remove bitrate for lossless
            options.pop("audio_bitrate", None)

        # Sample rate specific optimizations (simplified)
        if sample_rate <= 16000:
            # Low sample rate optimizations
            if format in ["mp3", "aac"]:
                # Use a conservative lowpass filter
                options["af"] = f"lowpass=f={sample_rate // 2 - 500}"
        elif sample_rate >= 44100:
            # High sample rate optimizations
            if format == "mp3":
                # Use a conservative highpass filter
                options["af"] = "highpass=f=20"

        return options


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
