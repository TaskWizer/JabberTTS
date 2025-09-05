"""Voice Cloning Engine for Advanced Voice Synthesis.

This module provides the core voice cloning engine that integrates
embedding extraction, voice modulation, and synthesis capabilities.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

from .embedding_extractor import VoiceEmbeddingExtractor, VoiceCharacteristics
from .voice_modulator import VoiceModulator, ModulationParameters
from .voice_library import VoiceLibraryManager, VoiceProfile
from ..inference.engine import InferenceEngine
from ..audio.processor import AudioProcessor

logger = logging.getLogger(__name__)


class VoiceCloningEngine:
    """Advanced voice cloning engine with real-time synthesis capabilities."""
    
    def __init__(self, library_path: Optional[Union[str, Path]] = None):
        """Initialize the voice cloning engine.
        
        Args:
            library_path: Path to voice library directory
        """
        self.library_path = library_path
        self.embedding_extractor = None
        self.voice_modulator = None
        self.voice_library = None
        self.inference_engine = None
        self.audio_processor = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the voice cloning engine."""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing voice cloning engine...")
            
            # Initialize components
            self.embedding_extractor = VoiceEmbeddingExtractor()
            await self.embedding_extractor.initialize()
            
            self.voice_modulator = VoiceModulator()
            await self.voice_modulator.initialize()
            
            self.voice_library = VoiceLibraryManager(self.library_path)
            await self.voice_library.initialize()
            
            # Get inference engine and audio processor from main system
            from ..inference.engine import get_inference_engine
            from ..audio.processor import get_audio_processor
            
            self.inference_engine = get_inference_engine()
            self.audio_processor = get_audio_processor()
            
            self.is_initialized = True
            logger.info("Voice cloning engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice cloning engine: {e}")
            raise
    
    async def clone_voice_from_sample(
        self,
        audio_sample: Union[bytes, np.ndarray, str, Path],
        voice_name: str,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """Clone a voice from an audio sample.
        
        Args:
            audio_sample: Audio sample for voice cloning
            voice_name: Name for the cloned voice
            description: Description of the voice
            tags: Tags for the voice
            
        Returns:
            Voice ID of the cloned voice
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            logger.info(f"Cloning voice '{voice_name}' from audio sample")
            
            # Add voice to library
            voice_id = await self.voice_library.add_voice_from_audio(
                name=voice_name,
                audio_data=audio_sample,
                description=description,
                tags=tags or [],
                source_type="cloned"
            )
            
            logger.info(f"Successfully cloned voice '{voice_name}' with ID {voice_id}")
            return voice_id
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise
    
    async def synthesize_with_cloned_voice(
        self,
        text: str,
        voice_id: str,
        modulation_params: Optional[ModulationParameters] = None,
        output_format: str = "mp3",
        speed: float = 1.0
    ) -> Tuple[bytes, Dict[str, any]]:
        """Synthesize speech using a cloned voice.
        
        Args:
            text: Text to synthesize
            voice_id: ID of the cloned voice
            modulation_params: Optional voice modulation parameters
            output_format: Output audio format
            speed: Speech speed
            
        Returns:
            Tuple of (audio data, metadata)
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get voice profile
            voice_profile = await self.voice_library.get_voice_profile(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice with ID {voice_id} not found")
            
            # Generate base audio using inference engine
            # Note: This is a simplified implementation
            # In production, you would integrate the voice characteristics
            # into the TTS model for actual voice cloning
            result = await self.inference_engine.generate_speech(
                text=text,
                voice="alloy",  # Base voice - would be replaced with cloned voice
                response_format="wav",
                speed=speed
            )
            
            # Apply voice modulation to simulate cloned voice characteristics
            audio_data = result["audio_data"]
            
            # Apply voice-specific modulation based on characteristics
            voice_modulation = self._create_voice_modulation(voice_profile.characteristics)
            
            # Combine with user-specified modulation
            if modulation_params:
                combined_modulation = self._combine_modulation_params(voice_modulation, modulation_params)
            else:
                combined_modulation = voice_modulation
            
            # Apply modulation
            modulated_audio = await self.voice_modulator.modulate_voice(
                audio_data, combined_modulation
            )
            
            # Process audio to final format
            processed_audio, audio_metadata = await self.audio_processor.process_audio(
                audio_array=modulated_audio,
                sample_rate=result["sample_rate"],
                output_format=output_format,
                speed=speed,
                original_sample_rate=result["sample_rate"]
            )
            
            # Update usage count
            await self.voice_library.increment_usage_count(voice_id)
            
            # Create response metadata
            response_metadata = {
                "voice_id": voice_id,
                "voice_name": voice_profile.name,
                "text_length": len(text),
                "audio_duration": audio_metadata["processed_duration"],
                "sample_rate": audio_metadata["final_sample_rate"],
                "format": output_format,
                "modulation_applied": combined_modulation is not None,
                "voice_characteristics": {
                    "fundamental_frequency": voice_profile.characteristics.fundamental_frequency,
                    "gender": voice_profile.gender,
                    "age_range": voice_profile.age_range,
                    "quality_score": voice_profile.quality_score
                }
            }
            
            return processed_audio, response_metadata
            
        except Exception as e:
            logger.error(f"Voice synthesis with cloned voice failed: {e}")
            raise
    
    def _create_voice_modulation(self, characteristics: VoiceCharacteristics) -> ModulationParameters:
        """Create modulation parameters based on voice characteristics."""
        try:
            # Convert voice characteristics to modulation parameters
            # This is a simplified mapping - production would use more sophisticated methods
            
            # Calculate pitch shift based on fundamental frequency
            base_f0 = 150.0  # Base frequency for comparison
            f0_ratio = characteristics.fundamental_frequency / base_f0 if base_f0 > 0 else 1.0
            pitch_shift = 12 * np.log2(f0_ratio) if f0_ratio > 0 else 0.0
            
            # Calculate formant shift based on spectral characteristics
            formant_shift = 1.0  # Default - would be calculated from formant analysis
            
            # Calculate voice effects based on quality metrics
            breathiness = characteristics.voice_quality_metrics.get('shimmer', 0.0) * 2.0
            roughness = characteristics.voice_quality_metrics.get('jitter', 0.0) * 5.0
            
            # Limit values to reasonable ranges
            pitch_shift = np.clip(pitch_shift, -12.0, 12.0)
            breathiness = np.clip(breathiness, 0.0, 1.0)
            roughness = np.clip(roughness, 0.0, 1.0)
            
            return ModulationParameters(
                pitch_shift=float(pitch_shift),
                formant_shift=formant_shift,
                voice_breathiness=float(breathiness),
                voice_roughness=float(roughness)
            )
            
        except Exception as e:
            logger.warning(f"Failed to create voice modulation parameters: {e}")
            return ModulationParameters()
    
    def _combine_modulation_params(
        self,
        voice_params: ModulationParameters,
        user_params: ModulationParameters
    ) -> ModulationParameters:
        """Combine voice-specific and user-specified modulation parameters."""
        return ModulationParameters(
            pitch_shift=voice_params.pitch_shift + user_params.pitch_shift,
            formant_shift=voice_params.formant_shift * user_params.formant_shift,
            speed_factor=user_params.speed_factor,  # User controls speed
            voice_breathiness=np.clip(voice_params.voice_breathiness + user_params.voice_breathiness, 0.0, 1.0),
            voice_roughness=np.clip(voice_params.voice_roughness + user_params.voice_roughness, 0.0, 1.0),
            resonance_boost=user_params.resonance_boost,
            harmonics_emphasis=user_params.harmonics_emphasis,
            gender_shift=user_params.gender_shift,
            age_shift=user_params.age_shift,
            emotion_intensity=user_params.emotion_intensity
        )
    
    async def analyze_voice_similarity(
        self,
        voice_id1: str,
        voice_id2: str
    ) -> Dict[str, float]:
        """Analyze similarity between two voices."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            voice1 = await self.voice_library.get_voice_profile(voice_id1)
            voice2 = await self.voice_library.get_voice_profile(voice_id2)
            
            if not voice1 or not voice2:
                raise ValueError("One or both voices not found")
            
            similarity_metrics = await self.embedding_extractor.compare_voices(
                voice1.characteristics,
                voice2.characteristics
            )
            
            return similarity_metrics
            
        except Exception as e:
            logger.error(f"Voice similarity analysis failed: {e}")
            raise
    
    async def get_voice_recommendations(
        self,
        text: str,
        preferences: Dict[str, any] = None
    ) -> List[Tuple[VoiceProfile, float]]:
        """Get voice recommendations for given text and preferences."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Get all available voices
            voices = await self.voice_library.list_voices()
            
            recommendations = []
            
            for voice in voices:
                score = self._calculate_voice_score(voice, text, preferences or {})
                recommendations.append((voice, score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:10]  # Return top 10 recommendations
            
        except Exception as e:
            logger.error(f"Voice recommendation failed: {e}")
            return []
    
    def _calculate_voice_score(
        self,
        voice: VoiceProfile,
        text: str,
        preferences: Dict[str, any]
    ) -> float:
        """Calculate recommendation score for a voice."""
        try:
            score = voice.quality_score  # Base score
            
            # Boost score based on preferences
            if preferences.get('gender') == voice.gender:
                score += 0.2
            
            if preferences.get('language') == voice.language:
                score += 0.3
            
            if preferences.get('category') == voice.category:
                score += 0.1
            
            # Boost popular voices
            score += min(voice.usage_count * 0.01, 0.2)
            
            # Boost favorites
            if voice.is_favorite:
                score += 0.1
            
            # Text-specific scoring (simplified)
            text_length = len(text)
            if text_length > 1000:  # Long text
                # Prefer high-quality voices for long text
                score += voice.quality_score * 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Voice scoring failed: {e}")
            return voice.quality_score
    
    async def create_voice_blend(
        self,
        voice_ids: List[str],
        weights: List[float],
        blend_name: str
    ) -> str:
        """Create a blended voice from multiple source voices."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if len(voice_ids) != len(weights):
                raise ValueError("Number of voice IDs must match number of weights")
            
            if abs(sum(weights) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")
            
            # Get voice profiles
            voices = []
            for voice_id in voice_ids:
                voice = await self.voice_library.get_voice_profile(voice_id)
                if not voice:
                    raise ValueError(f"Voice {voice_id} not found")
                voices.append(voice)
            
            # Blend voice characteristics
            blended_characteristics = self._blend_voice_characteristics(voices, weights)
            
            # Create blended voice profile
            # This is a simplified implementation - production would create actual blended audio
            blend_id = await self.voice_library.add_voice_from_audio(
                name=blend_name,
                audio_data=voices[0].audio_samples[0],  # Use first voice's sample as base
                description=f"Blended voice from {len(voices)} sources",
                tags=["blended", "custom"],
                category="blended",
                creator="system"
            )
            
            # Update characteristics with blended values
            blend_profile = await self.voice_library.get_voice_profile(blend_id)
            blend_profile.characteristics = blended_characteristics
            blend_profile.source_type = "blended"
            
            await self.voice_library._save_voice_profile(blend_profile)
            
            return blend_id
            
        except Exception as e:
            logger.error(f"Voice blending failed: {e}")
            raise
    
    def _blend_voice_characteristics(
        self,
        voices: List[VoiceProfile],
        weights: List[float]
    ) -> VoiceCharacteristics:
        """Blend voice characteristics from multiple voices."""
        try:
            # Weighted average of embeddings
            blended_embedding = np.zeros_like(voices[0].characteristics.embedding)
            for voice, weight in zip(voices, weights):
                blended_embedding += voice.characteristics.embedding * weight
            
            # Weighted average of other characteristics
            blended_f0 = sum(voice.characteristics.fundamental_frequency * weight 
                           for voice, weight in zip(voices, weights))
            
            blended_formants = []
            for i in range(len(voices[0].characteristics.formant_frequencies)):
                blended_formant = sum(voice.characteristics.formant_frequencies[i] * weight
                                    for voice, weight in zip(voices, weights))
                blended_formants.append(blended_formant)
            
            # Use first voice's characteristics as base and update key values
            base_characteristics = voices[0].characteristics
            
            return VoiceCharacteristics(
                embedding=blended_embedding,
                fundamental_frequency=blended_f0,
                formant_frequencies=blended_formants,
                spectral_centroid=base_characteristics.spectral_centroid,
                spectral_rolloff=base_characteristics.spectral_rolloff,
                zero_crossing_rate=base_characteristics.zero_crossing_rate,
                mfcc_features=base_characteristics.mfcc_features,
                pitch_range=base_characteristics.pitch_range,
                voice_quality_metrics=base_characteristics.voice_quality_metrics,
                confidence_score=np.mean([voice.characteristics.confidence_score for voice in voices])
            )
            
        except Exception as e:
            logger.error(f"Voice characteristics blending failed: {e}")
            return voices[0].characteristics
