"""Voice Modulator for Real-time Voice Manipulation.

This module provides comprehensive voice modulation capabilities including
pitch shifting, formant manipulation, speed control, and voice effects.
"""

import asyncio
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import signal
import scipy.ndimage

logger = logging.getLogger(__name__)


@dataclass
class ModulationParameters:
    """Parameters for voice modulation."""
    pitch_shift: float = 0.0  # Semitones (-12 to +12)
    formant_shift: float = 0.0  # Ratio (0.5 to 2.0)
    speed_factor: float = 1.0  # Speed multiplier (0.5 to 2.0)
    voice_breathiness: float = 0.0  # Breathiness amount (0.0 to 1.0)
    voice_roughness: float = 0.0  # Roughness amount (0.0 to 1.0)
    resonance_boost: float = 0.0  # Resonance enhancement (-1.0 to 1.0)
    harmonics_emphasis: float = 0.0  # Harmonic emphasis (-1.0 to 1.0)
    gender_shift: float = 0.0  # Gender transformation (-1.0 to 1.0)
    age_shift: float = 0.0  # Age transformation (-1.0 to 1.0)
    emotion_intensity: float = 0.0  # Emotional intensity (0.0 to 1.0)


class VoiceModulator:
    """Advanced voice modulator for real-time voice manipulation."""
    
    def __init__(self, sample_rate: int = 24000):
        """Initialize the voice modulator.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the voice modulator."""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing voice modulator...")
            
            # Initialize processing components
            self.pitch_shifter = PitchShifter(self.sample_rate)
            self.formant_shifter = FormantShifter(self.sample_rate)
            self.speed_controller = SpeedController(self.sample_rate)
            self.voice_effects = VoiceEffects(self.sample_rate)
            
            self.is_initialized = True
            logger.info("Voice modulator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice modulator: {e}")
            raise
    
    async def modulate_voice(
        self, 
        audio_data: np.ndarray, 
        parameters: ModulationParameters
    ) -> np.ndarray:
        """Apply voice modulation with specified parameters.
        
        Args:
            audio_data: Input audio data
            parameters: Modulation parameters
            
        Returns:
            Modulated audio data
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            modulated_audio = audio_data.copy()
            
            # Apply pitch shifting
            if abs(parameters.pitch_shift) > 0.01:
                modulated_audio = await self.pitch_shifter.shift_pitch(
                    modulated_audio, parameters.pitch_shift
                )
            
            # Apply formant shifting
            if abs(parameters.formant_shift) > 0.01:
                modulated_audio = await self.formant_shifter.shift_formants(
                    modulated_audio, parameters.formant_shift
                )
            
            # Apply speed control
            if abs(parameters.speed_factor - 1.0) > 0.01:
                modulated_audio = await self.speed_controller.adjust_speed(
                    modulated_audio, parameters.speed_factor
                )
            
            # Apply voice effects
            modulated_audio = await self.voice_effects.apply_effects(
                modulated_audio, parameters
            )
            
            # Apply gender and age transformations
            if abs(parameters.gender_shift) > 0.01 or abs(parameters.age_shift) > 0.01:
                modulated_audio = await self._apply_demographic_transformations(
                    modulated_audio, parameters
                )
            
            return modulated_audio
            
        except Exception as e:
            logger.error(f"Voice modulation failed: {e}")
            raise
    
    async def _apply_demographic_transformations(
        self, 
        audio_data: np.ndarray, 
        parameters: ModulationParameters
    ) -> np.ndarray:
        """Apply gender and age transformations."""
        try:
            transformed_audio = audio_data.copy()
            
            # Gender transformation (simplified)
            if abs(parameters.gender_shift) > 0.01:
                # Positive values: more feminine (higher pitch, shifted formants)
                # Negative values: more masculine (lower pitch, shifted formants)
                gender_pitch_shift = parameters.gender_shift * 4.0  # Up to 4 semitones
                gender_formant_shift = 1.0 + (parameters.gender_shift * 0.2)  # Up to 20% formant shift
                
                transformed_audio = await self.pitch_shifter.shift_pitch(
                    transformed_audio, gender_pitch_shift
                )
                transformed_audio = await self.formant_shifter.shift_formants(
                    transformed_audio, gender_formant_shift
                )
            
            # Age transformation (simplified)
            if abs(parameters.age_shift) > 0.01:
                # Positive values: older (more breathiness, different formants)
                # Negative values: younger (clearer voice, different resonance)
                age_breathiness = max(0.0, parameters.age_shift * 0.3)
                age_roughness = max(0.0, parameters.age_shift * 0.2)
                
                # Apply age-related voice effects
                age_params = ModulationParameters(
                    voice_breathiness=age_breathiness,
                    voice_roughness=age_roughness,
                    resonance_boost=-parameters.age_shift * 0.1
                )
                transformed_audio = await self.voice_effects.apply_effects(
                    transformed_audio, age_params
                )
            
            return transformed_audio
            
        except Exception as e:
            logger.error(f"Demographic transformation failed: {e}")
            return audio_data
    
    async def create_voice_preset(
        self, 
        name: str, 
        parameters: ModulationParameters
    ) -> Dict[str, any]:
        """Create a voice modulation preset.
        
        Args:
            name: Preset name
            parameters: Modulation parameters
            
        Returns:
            Preset dictionary
        """
        return {
            "name": name,
            "parameters": {
                "pitch_shift": parameters.pitch_shift,
                "formant_shift": parameters.formant_shift,
                "speed_factor": parameters.speed_factor,
                "voice_breathiness": parameters.voice_breathiness,
                "voice_roughness": parameters.voice_roughness,
                "resonance_boost": parameters.resonance_boost,
                "harmonics_emphasis": parameters.harmonics_emphasis,
                "gender_shift": parameters.gender_shift,
                "age_shift": parameters.age_shift,
                "emotion_intensity": parameters.emotion_intensity
            },
            "created_at": asyncio.get_event_loop().time()
        }


class PitchShifter:
    """Pitch shifting component."""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    async def shift_pitch(self, audio_data: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by specified semitones."""
        try:
            if abs(semitones) < 0.01:
                return audio_data
            
            # Use librosa's pitch shifting
            shifted_audio = librosa.effects.pitch_shift(
                audio_data, 
                sr=self.sample_rate, 
                n_steps=semitones
            )
            
            return shifted_audio
            
        except Exception as e:
            logger.error(f"Pitch shifting failed: {e}")
            return audio_data


class FormantShifter:
    """Formant shifting component."""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    async def shift_formants(self, audio_data: np.ndarray, ratio: float) -> np.ndarray:
        """Shift formants by specified ratio."""
        try:
            if abs(ratio - 1.0) < 0.01:
                return audio_data
            
            # Simplified formant shifting using spectral envelope manipulation
            # This is a basic implementation - production would use more sophisticated methods
            
            # Compute STFT
            stft = librosa.stft(audio_data, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Shift spectral envelope (simplified formant shifting)
            shifted_magnitude = self._shift_spectral_envelope(magnitude, ratio)
            
            # Reconstruct audio
            shifted_stft = shifted_magnitude * np.exp(1j * phase)
            shifted_audio = librosa.istft(shifted_stft, hop_length=512)
            
            return shifted_audio
            
        except Exception as e:
            logger.error(f"Formant shifting failed: {e}")
            return audio_data
    
    def _shift_spectral_envelope(self, magnitude: np.ndarray, ratio: float) -> np.ndarray:
        """Shift spectral envelope to simulate formant shifting."""
        try:
            # This is a simplified implementation
            # Real formant shifting requires more sophisticated spectral envelope manipulation
            
            freq_bins, time_frames = magnitude.shape
            shifted_magnitude = np.zeros_like(magnitude)
            
            for t in range(time_frames):
                spectrum = magnitude[:, t]
                
                # Create frequency axis
                freqs = np.linspace(0, self.sample_rate // 2, freq_bins)
                
                # Shift frequencies
                shifted_freqs = freqs * ratio
                
                # Interpolate to get shifted spectrum
                shifted_spectrum = np.interp(freqs, shifted_freqs, spectrum)
                shifted_magnitude[:, t] = shifted_spectrum
            
            return shifted_magnitude
            
        except Exception as e:
            logger.error(f"Spectral envelope shifting failed: {e}")
            return magnitude


class SpeedController:
    """Speed control component."""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    async def adjust_speed(self, audio_data: np.ndarray, factor: float) -> np.ndarray:
        """Adjust playback speed by specified factor."""
        try:
            if abs(factor - 1.0) < 0.01:
                return audio_data
            
            # Use librosa's time stretching
            stretched_audio = librosa.effects.time_stretch(audio_data, rate=factor)
            
            return stretched_audio
            
        except Exception as e:
            logger.error(f"Speed adjustment failed: {e}")
            return audio_data


class VoiceEffects:
    """Voice effects component."""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
    
    async def apply_effects(
        self, 
        audio_data: np.ndarray, 
        parameters: ModulationParameters
    ) -> np.ndarray:
        """Apply various voice effects."""
        try:
            processed_audio = audio_data.copy()
            
            # Apply breathiness
            if parameters.voice_breathiness > 0.01:
                processed_audio = self._add_breathiness(processed_audio, parameters.voice_breathiness)
            
            # Apply roughness
            if parameters.voice_roughness > 0.01:
                processed_audio = self._add_roughness(processed_audio, parameters.voice_roughness)
            
            # Apply resonance boost
            if abs(parameters.resonance_boost) > 0.01:
                processed_audio = self._adjust_resonance(processed_audio, parameters.resonance_boost)
            
            # Apply harmonics emphasis
            if abs(parameters.harmonics_emphasis) > 0.01:
                processed_audio = self._adjust_harmonics(processed_audio, parameters.harmonics_emphasis)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Voice effects application failed: {e}")
            return audio_data
    
    def _add_breathiness(self, audio_data: np.ndarray, amount: float) -> np.ndarray:
        """Add breathiness to voice."""
        try:
            # Add filtered noise to simulate breathiness
            noise = np.random.randn(len(audio_data)) * 0.1 * amount
            
            # Filter noise to match voice characteristics
            b, a = signal.butter(4, 0.3, btype='high')
            filtered_noise = signal.filtfilt(b, a, noise)
            
            # Mix with original audio
            breathy_audio = audio_data + filtered_noise * amount
            
            return breathy_audio
            
        except Exception as e:
            logger.error(f"Breathiness addition failed: {e}")
            return audio_data
    
    def _add_roughness(self, audio_data: np.ndarray, amount: float) -> np.ndarray:
        """Add roughness to voice."""
        try:
            # Add amplitude modulation to simulate roughness
            modulation_freq = 30.0  # Hz
            t = np.arange(len(audio_data)) / self.sample_rate
            modulation = 1.0 + amount * 0.3 * np.sin(2 * np.pi * modulation_freq * t)
            
            rough_audio = audio_data * modulation
            
            return rough_audio
            
        except Exception as e:
            logger.error(f"Roughness addition failed: {e}")
            return audio_data
    
    def _adjust_resonance(self, audio_data: np.ndarray, boost: float) -> np.ndarray:
        """Adjust voice resonance."""
        try:
            # Apply resonant filtering
            if boost > 0:
                # Boost resonance with peaking filter
                freq = 1000.0  # Hz
                Q = 2.0
                gain = boost * 6.0  # dB
                
                # Design peaking filter
                w0 = 2 * np.pi * freq / self.sample_rate
                A = 10 ** (gain / 40)
                alpha = np.sin(w0) / (2 * Q)
                
                b0 = 1 + alpha * A
                b1 = -2 * np.cos(w0)
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * np.cos(w0)
                a2 = 1 - alpha / A
                
                b = [b0/a0, b1/a0, b2/a0]
                a = [1, a1/a0, a2/a0]
                
                resonant_audio = signal.filtfilt(b, a, audio_data)
            else:
                # Reduce resonance with notch filter
                resonant_audio = audio_data  # Simplified - would implement notch filter
            
            return resonant_audio
            
        except Exception as e:
            logger.error(f"Resonance adjustment failed: {e}")
            return audio_data
    
    def _adjust_harmonics(self, audio_data: np.ndarray, emphasis: float) -> np.ndarray:
        """Adjust harmonic emphasis."""
        try:
            # Apply harmonic enhancement/reduction
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Enhance or reduce harmonics (simplified)
            if emphasis > 0:
                # Enhance harmonics by boosting higher frequencies
                freq_weights = np.linspace(1.0, 1.0 + emphasis, magnitude.shape[0])
                enhanced_magnitude = magnitude * freq_weights[:, np.newaxis]
            else:
                # Reduce harmonics by attenuating higher frequencies
                freq_weights = np.linspace(1.0, 1.0 + emphasis, magnitude.shape[0])
                enhanced_magnitude = magnitude * freq_weights[:, np.newaxis]
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Harmonics adjustment failed: {e}")
            return audio_data
