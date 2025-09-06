"""Advanced Speed Control with High-Quality Time-Stretching for JabberTTS.

This module implements proper time-stretching algorithms to fix speed control distortion
issues, replacing simple sample rate manipulation with sophisticated algorithms that
maintain audio fidelity and naturalness across the full speed range (0.25x-4.0x).
"""

import logging
import numpy as np
from typing import Optional, Union, Tuple
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class TimeStretchAlgorithm(Enum):
    """Available time-stretching algorithms."""
    LIBROSA_PHASE_VOCODER = "librosa_phase_vocoder"
    LIBROSA_TIME_STRETCH = "librosa_time_stretch"
    PSOLA = "psola"
    WSOLA = "wsola"
    SIMPLE_OVERLAP_ADD = "simple_overlap_add"
    FALLBACK_RESAMPLE = "fallback_resample"


class AdvancedSpeedController:
    """Advanced speed control with multiple time-stretching algorithms."""
    
    def __init__(self, algorithm: TimeStretchAlgorithm = TimeStretchAlgorithm.LIBROSA_TIME_STRETCH):
        """Initialize the speed controller.
        
        Args:
            algorithm: Time-stretching algorithm to use
        """
        self.algorithm = algorithm
        self._librosa_available = self._check_librosa_availability()
        self._scipy_available = self._check_scipy_availability()
        
        # Algorithm fallback chain
        self.fallback_chain = [
            TimeStretchAlgorithm.LIBROSA_TIME_STRETCH,
            TimeStretchAlgorithm.LIBROSA_PHASE_VOCODER,
            TimeStretchAlgorithm.SIMPLE_OVERLAP_ADD,
            TimeStretchAlgorithm.FALLBACK_RESAMPLE
        ]
        
        logger.info(f"Initialized AdvancedSpeedController with {algorithm.value}")
    
    def adjust_speed(
        self,
        audio: np.ndarray,
        speed_factor: float,
        sample_rate: int,
        preserve_pitch: bool = True
    ) -> np.ndarray:
        """Adjust audio speed using advanced time-stretching.
        
        Args:
            audio: Input audio array
            speed_factor: Speed multiplication factor (0.25-4.0)
            sample_rate: Audio sample rate
            preserve_pitch: Whether to preserve pitch during speed change
            
        Returns:
            Speed-adjusted audio array
        """
        # Validate inputs
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be a numpy array")
        
        if len(audio) == 0:
            return audio
        
        if speed_factor <= 0:
            raise ValueError("Speed factor must be positive")
        
        # No change needed
        if abs(speed_factor - 1.0) < 0.001:
            return audio
        
        # Clamp speed factor to reasonable range
        speed_factor = max(0.25, min(4.0, speed_factor))
        
        logger.debug(f"Adjusting speed by factor {speed_factor:.3f} using {self.algorithm.value}")
        
        try:
            # Try primary algorithm
            return self._apply_algorithm(audio, speed_factor, sample_rate, preserve_pitch, self.algorithm)
        
        except Exception as e:
            logger.warning(f"Primary algorithm {self.algorithm.value} failed: {e}")
            
            # Try fallback algorithms
            for fallback_algo in self.fallback_chain:
                if fallback_algo == self.algorithm:
                    continue
                
                try:
                    logger.info(f"Trying fallback algorithm: {fallback_algo.value}")
                    return self._apply_algorithm(audio, speed_factor, sample_rate, preserve_pitch, fallback_algo)
                
                except Exception as fallback_error:
                    logger.warning(f"Fallback algorithm {fallback_algo.value} failed: {fallback_error}")
                    continue
            
            # All algorithms failed, return original audio
            logger.error("All time-stretching algorithms failed, returning original audio")
            return audio
    
    def _apply_algorithm(
        self,
        audio: np.ndarray,
        speed_factor: float,
        sample_rate: int,
        preserve_pitch: bool,
        algorithm: TimeStretchAlgorithm
    ) -> np.ndarray:
        """Apply specific time-stretching algorithm."""
        
        if algorithm == TimeStretchAlgorithm.LIBROSA_TIME_STRETCH:
            return self._librosa_time_stretch(audio, speed_factor, sample_rate)
        
        elif algorithm == TimeStretchAlgorithm.LIBROSA_PHASE_VOCODER:
            return self._librosa_phase_vocoder(audio, speed_factor, sample_rate)
        
        elif algorithm == TimeStretchAlgorithm.PSOLA:
            return self._psola_time_stretch(audio, speed_factor, sample_rate)
        
        elif algorithm == TimeStretchAlgorithm.WSOLA:
            return self._wsola_time_stretch(audio, speed_factor, sample_rate)
        
        elif algorithm == TimeStretchAlgorithm.SIMPLE_OVERLAP_ADD:
            return self._simple_overlap_add(audio, speed_factor, sample_rate)
        
        elif algorithm == TimeStretchAlgorithm.FALLBACK_RESAMPLE:
            return self._fallback_resample(audio, speed_factor, sample_rate)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _librosa_time_stretch(self, audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """High-quality time-stretching using librosa."""
        if not self._librosa_available:
            raise ImportError("librosa not available")
        
        import librosa
        
        # librosa.effects.time_stretch expects rate parameter (inverse of speed_factor)
        rate = speed_factor
        
        try:
            # Use librosa's time_stretch which preserves pitch
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            
            # Ensure output is finite and within valid range
            stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=-1.0)
            stretched = np.clip(stretched, -1.0, 1.0)
            
            return stretched.astype(audio.dtype)
            
        except Exception as e:
            raise RuntimeError(f"librosa time_stretch failed: {e}")
    
    def _librosa_phase_vocoder(self, audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """Phase vocoder time-stretching using librosa."""
        if not self._librosa_available:
            raise ImportError("librosa not available")
        
        import librosa
        
        try:
            # Convert to STFT
            hop_length = 512
            stft = librosa.stft(audio, hop_length=hop_length)
            
            # Apply phase vocoder
            stft_stretched = librosa.phase_vocoder(stft, rate=speed_factor, hop_length=hop_length)
            
            # Convert back to time domain
            stretched = librosa.istft(stft_stretched, hop_length=hop_length)
            
            # Ensure output is finite and within valid range
            stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=-1.0)
            stretched = np.clip(stretched, -1.0, 1.0)
            
            return stretched.astype(audio.dtype)
            
        except Exception as e:
            raise RuntimeError(f"librosa phase_vocoder failed: {e}")
    
    def _psola_time_stretch(self, audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """PSOLA (Pitch Synchronous Overlap and Add) time-stretching."""
        # Simplified PSOLA implementation
        # This is a basic version - a full PSOLA implementation would require pitch detection
        
        try:
            # Use overlap-add with pitch-based frame sizing
            frame_size = int(0.02 * sample_rate)  # 20ms frames
            hop_size = int(frame_size / speed_factor)
            
            if hop_size <= 0:
                hop_size = 1
            
            # Generate output
            output_length = int(len(audio) / speed_factor)
            output = np.zeros(output_length, dtype=audio.dtype)
            
            # Overlap-add with Hanning window
            window = np.hanning(frame_size)
            
            input_pos = 0
            output_pos = 0
            
            while input_pos + frame_size < len(audio) and output_pos + frame_size < len(output):
                # Extract frame
                frame = audio[input_pos:input_pos + frame_size] * window
                
                # Add to output with overlap
                end_pos = min(output_pos + frame_size, len(output))
                frame_end = end_pos - output_pos
                output[output_pos:end_pos] += frame[:frame_end]
                
                input_pos += hop_size
                output_pos += int(hop_size / speed_factor)
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = output / max_val * 0.95
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"PSOLA time_stretch failed: {e}")
    
    def _wsola_time_stretch(self, audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """WSOLA (Waveform Similarity Overlap-Add) time-stretching."""
        try:
            # WSOLA parameters
            frame_size = int(0.02 * sample_rate)  # 20ms
            overlap_size = frame_size // 2
            hop_size = int(frame_size / speed_factor)
            
            if hop_size <= 0:
                hop_size = 1
            
            output_length = int(len(audio) / speed_factor)
            output = np.zeros(output_length, dtype=audio.dtype)
            
            # Hanning window for overlap
            window = np.hanning(frame_size)
            
            input_pos = 0
            output_pos = 0
            
            while input_pos + frame_size < len(audio) and output_pos + frame_size < len(output):
                # Extract current frame
                frame = audio[input_pos:input_pos + frame_size]
                
                # Find best overlap position (simplified correlation)
                if output_pos > overlap_size:
                    # Cross-correlation for best overlap
                    search_range = min(overlap_size, len(audio) - input_pos - frame_size)
                    best_offset = 0
                    best_correlation = -1
                    
                    for offset in range(-search_range//2, search_range//2):
                        if input_pos + offset >= 0 and input_pos + offset + frame_size < len(audio):
                            test_frame = audio[input_pos + offset:input_pos + offset + frame_size]
                            overlap_region = output[output_pos:output_pos + overlap_size]
                            
                            if len(overlap_region) == overlap_size and len(test_frame) >= overlap_size:
                                correlation = np.corrcoef(overlap_region, test_frame[:overlap_size])[0, 1]
                                if not np.isnan(correlation) and correlation > best_correlation:
                                    best_correlation = correlation
                                    best_offset = offset
                    
                    input_pos += best_offset
                    frame = audio[input_pos:input_pos + frame_size]
                
                # Apply window and add to output
                windowed_frame = frame * window
                
                end_pos = min(output_pos + frame_size, len(output))
                frame_end = end_pos - output_pos
                output[output_pos:end_pos] += windowed_frame[:frame_end]
                
                input_pos += hop_size
                output_pos += int(hop_size / speed_factor)
            
            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = output / max_val * 0.95
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"WSOLA time_stretch failed: {e}")
    
    def _simple_overlap_add(self, audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """Simple overlap-add time-stretching."""
        try:
            frame_size = int(0.025 * sample_rate)  # 25ms frames
            hop_input = int(frame_size * 0.75)  # 75% overlap
            hop_output = int(hop_input / speed_factor)
            
            if hop_output <= 0:
                hop_output = 1
            
            output_length = int(len(audio) / speed_factor)
            output = np.zeros(output_length, dtype=audio.dtype)
            
            # Hanning window
            window = np.hanning(frame_size)
            
            input_pos = 0
            output_pos = 0
            
            while input_pos + frame_size < len(audio) and output_pos + frame_size < len(output):
                # Extract and window frame
                frame = audio[input_pos:input_pos + frame_size] * window
                
                # Add to output
                end_pos = min(output_pos + frame_size, len(output))
                frame_end = end_pos - output_pos
                output[output_pos:end_pos] += frame[:frame_end]
                
                input_pos += hop_input
                output_pos += hop_output
            
            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 1.0:
                output = output / max_val * 0.95
            
            return output
            
        except Exception as e:
            raise RuntimeError(f"Simple overlap-add failed: {e}")
    
    def _fallback_resample(self, audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
        """Fallback method using simple resampling (changes pitch)."""
        try:
            if self._scipy_available:
                from scipy import signal
                
                # Calculate new sample rate
                new_sample_rate = int(sample_rate * speed_factor)
                
                # Resample
                resampled = signal.resample(audio, int(len(audio) / speed_factor))
                
                return resampled.astype(audio.dtype)
            else:
                # Very basic linear interpolation fallback
                new_length = int(len(audio) / speed_factor)
                indices = np.linspace(0, len(audio) - 1, new_length)
                
                # Linear interpolation
                output = np.interp(indices, np.arange(len(audio)), audio)
                
                return output.astype(audio.dtype)
                
        except Exception as e:
            raise RuntimeError(f"Fallback resample failed: {e}")
    
    def _check_librosa_availability(self) -> bool:
        """Check if librosa is available."""
        try:
            import librosa
            return True
        except ImportError:
            return False
    
    def _check_scipy_availability(self) -> bool:
        """Check if scipy is available."""
        try:
            import scipy
            return True
        except ImportError:
            return False
    
    def get_algorithm_info(self) -> dict:
        """Get information about available algorithms."""
        return {
            "current_algorithm": self.algorithm.value,
            "librosa_available": self._librosa_available,
            "scipy_available": self._scipy_available,
            "fallback_chain": [algo.value for algo in self.fallback_chain]
        }


# Global instance
_speed_controller = None


def get_speed_controller(algorithm: Optional[TimeStretchAlgorithm] = None) -> AdvancedSpeedController:
    """Get the global speed controller instance."""
    global _speed_controller
    if _speed_controller is None or (algorithm and algorithm != _speed_controller.algorithm):
        _speed_controller = AdvancedSpeedController(algorithm or TimeStretchAlgorithm.LIBROSA_TIME_STRETCH)
    return _speed_controller


def adjust_audio_speed(
    audio: np.ndarray,
    speed_factor: float,
    sample_rate: int,
    algorithm: Optional[TimeStretchAlgorithm] = None,
    preserve_pitch: bool = True
) -> np.ndarray:
    """Convenience function to adjust audio speed.
    
    Args:
        audio: Input audio array
        speed_factor: Speed multiplication factor (0.25-4.0)
        sample_rate: Audio sample rate
        algorithm: Time-stretching algorithm to use
        preserve_pitch: Whether to preserve pitch during speed change
        
    Returns:
        Speed-adjusted audio array
    """
    controller = get_speed_controller(algorithm)
    return controller.adjust_speed(audio, speed_factor, sample_rate, preserve_pitch)
