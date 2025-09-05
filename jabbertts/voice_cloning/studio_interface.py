"""Voice Cloning Studio Interface.

This module provides the main interface for the Voice Cloning Studio,
integrating all voice cloning capabilities into a cohesive system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

from .cloning_engine import VoiceCloningEngine
from .voice_modulator import ModulationParameters
from .voice_library import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceCloningStudio:
    """Main interface for the Voice Cloning Studio."""
    
    def __init__(self, library_path: Optional[Union[str, Path]] = None):
        """Initialize the Voice Cloning Studio.
        
        Args:
            library_path: Path to voice library directory
        """
        self.cloning_engine = VoiceCloningEngine(library_path)
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the Voice Cloning Studio."""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing Voice Cloning Studio...")
            await self.cloning_engine.initialize()
            self.is_initialized = True
            logger.info("Voice Cloning Studio initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Voice Cloning Studio: {e}")
            raise
    
    # Voice Management Methods
    
    async def upload_voice_sample(
        self,
        audio_data: Union[bytes, str, Path],
        voice_name: str,
        description: str = "",
        tags: List[str] = None,
        category: str = "custom"
    ) -> Dict[str, Any]:
        """Upload and process a voice sample for cloning.
        
        Args:
            audio_data: Audio data (bytes, file path, or URL)
            voice_name: Name for the voice
            description: Voice description
            tags: Voice tags
            category: Voice category
            
        Returns:
            Dictionary with voice ID and analysis results
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Clone voice from sample
            voice_id = await self.cloning_engine.clone_voice_from_sample(
                audio_sample=audio_data,
                voice_name=voice_name,
                description=description,
                tags=tags or []
            )
            
            # Get voice profile for analysis results
            voice_profile = await self.cloning_engine.voice_library.get_voice_profile(voice_id)
            
            return {
                "success": True,
                "voice_id": voice_id,
                "voice_name": voice_name,
                "analysis": {
                    "fundamental_frequency": voice_profile.characteristics.fundamental_frequency,
                    "gender": voice_profile.gender,
                    "age_range": voice_profile.age_range,
                    "quality_score": voice_profile.quality_score,
                    "confidence_score": voice_profile.characteristics.confidence_score,
                    "formant_frequencies": voice_profile.characteristics.formant_frequencies,
                    "voice_quality_metrics": voice_profile.characteristics.voice_quality_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Voice sample upload failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_voice_preview(
        self,
        voice_id: str,
        preview_text: str = "Hello, this is a preview of my voice.",
        modulation_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate a preview of a cloned voice.
        
        Args:
            voice_id: ID of the voice to preview
            preview_text: Text for the preview
            modulation_params: Optional modulation parameters
            
        Returns:
            Dictionary with audio data and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Convert modulation parameters
            mod_params = None
            if modulation_params:
                mod_params = ModulationParameters(
                    pitch_shift=modulation_params.get('pitch_shift', 0.0),
                    formant_shift=modulation_params.get('formant_shift', 1.0),
                    speed_factor=modulation_params.get('speed_factor', 1.0),
                    voice_breathiness=modulation_params.get('voice_breathiness', 0.0),
                    voice_roughness=modulation_params.get('voice_roughness', 0.0),
                    resonance_boost=modulation_params.get('resonance_boost', 0.0),
                    harmonics_emphasis=modulation_params.get('harmonics_emphasis', 0.0),
                    gender_shift=modulation_params.get('gender_shift', 0.0),
                    age_shift=modulation_params.get('age_shift', 0.0),
                    emotion_intensity=modulation_params.get('emotion_intensity', 0.0)
                )
            
            # Generate speech with cloned voice
            audio_data, metadata = await self.cloning_engine.synthesize_with_cloned_voice(
                text=preview_text,
                voice_id=voice_id,
                modulation_params=mod_params,
                output_format="mp3"
            )
            
            return {
                "success": True,
                "audio_data": audio_data,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Voice preview generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_voices(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """List available voices with optional filtering.
        
        Args:
            filters: Optional filters (category, language, gender, tags, favorites_only)
            
        Returns:
            Dictionary with voice list and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            filters = filters or {}
            
            voices = await self.cloning_engine.voice_library.list_voices(
                category=filters.get('category'),
                language=filters.get('language'),
                gender=filters.get('gender'),
                tags=filters.get('tags'),
                favorites_only=filters.get('favorites_only', False)
            )
            
            # Convert to serializable format
            voice_list = []
            for voice in voices:
                voice_list.append({
                    "id": voice.id,
                    "name": voice.name,
                    "description": voice.description,
                    "category": voice.category,
                    "language": voice.language,
                    "gender": voice.gender,
                    "age_range": voice.age_range,
                    "quality_score": voice.quality_score,
                    "usage_count": voice.usage_count,
                    "is_favorite": voice.is_favorite,
                    "tags": voice.tags,
                    "created_at": voice.created_at,
                    "source_type": voice.source_type
                })
            
            return {
                "success": True,
                "voices": voice_list,
                "total_count": len(voice_list)
            }
            
        except Exception as e:
            logger.error(f"Voice listing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_voices(self, query: str) -> Dict[str, Any]:
        """Search voices by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            voices = await self.cloning_engine.voice_library.search_voices(query)
            
            # Convert to serializable format
            voice_list = []
            for voice in voices:
                voice_list.append({
                    "id": voice.id,
                    "name": voice.name,
                    "description": voice.description,
                    "category": voice.category,
                    "language": voice.language,
                    "gender": voice.gender,
                    "quality_score": voice.quality_score,
                    "tags": voice.tags
                })
            
            return {
                "success": True,
                "voices": voice_list,
                "query": query,
                "result_count": len(voice_list)
            }
            
        except Exception as e:
            logger.error(f"Voice search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_voice_details(self, voice_id: str) -> Dict[str, Any]:
        """Get detailed information about a voice.
        
        Args:
            voice_id: Voice ID
            
        Returns:
            Dictionary with detailed voice information
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            voice = await self.cloning_engine.voice_library.get_voice_profile(voice_id)
            if not voice:
                return {
                    "success": False,
                    "error": "Voice not found"
                }
            
            return {
                "success": True,
                "voice": {
                    "id": voice.id,
                    "name": voice.name,
                    "description": voice.description,
                    "category": voice.category,
                    "language": voice.language,
                    "gender": voice.gender,
                    "age_range": voice.age_range,
                    "quality_score": voice.quality_score,
                    "usage_count": voice.usage_count,
                    "is_favorite": voice.is_favorite,
                    "tags": voice.tags,
                    "created_at": voice.created_at,
                    "updated_at": voice.updated_at,
                    "source_type": voice.source_type,
                    "creator": voice.creator,
                    "characteristics": {
                        "fundamental_frequency": voice.characteristics.fundamental_frequency,
                        "formant_frequencies": voice.characteristics.formant_frequencies,
                        "spectral_centroid": voice.characteristics.spectral_centroid,
                        "spectral_rolloff": voice.characteristics.spectral_rolloff,
                        "zero_crossing_rate": voice.characteristics.zero_crossing_rate,
                        "pitch_range": voice.characteristics.pitch_range,
                        "voice_quality_metrics": voice.characteristics.voice_quality_metrics,
                        "confidence_score": voice.characteristics.confidence_score
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Get voice details failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_voice_metadata(
        self,
        voice_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update voice metadata.
        
        Args:
            voice_id: Voice ID
            updates: Dictionary with fields to update
            
        Returns:
            Dictionary with update result
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            success = await self.cloning_engine.voice_library.update_voice_metadata(
                voice_id=voice_id,
                name=updates.get('name'),
                description=updates.get('description'),
                tags=updates.get('tags'),
                category=updates.get('category'),
                is_favorite=updates.get('is_favorite')
            )
            
            return {
                "success": success,
                "voice_id": voice_id
            }
            
        except Exception as e:
            logger.error(f"Voice metadata update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_voice(self, voice_id: str) -> Dict[str, Any]:
        """Delete a voice from the library.
        
        Args:
            voice_id: Voice ID
            
        Returns:
            Dictionary with deletion result
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            success = await self.cloning_engine.voice_library.delete_voice(voice_id)
            
            return {
                "success": success,
                "voice_id": voice_id
            }
            
        except Exception as e:
            logger.error(f"Voice deletion failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Advanced Features
    
    async def find_similar_voices(
        self,
        voice_id: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Find voices similar to the specified voice.
        
        Args:
            voice_id: Reference voice ID
            max_results: Maximum number of results
            
        Returns:
            Dictionary with similar voices
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            similar_voices = await self.cloning_engine.voice_library.find_similar_voices(
                voice_id, max_results
            )
            
            # Convert to serializable format
            results = []
            for voice, similarity in similar_voices:
                results.append({
                    "voice": {
                        "id": voice.id,
                        "name": voice.name,
                        "description": voice.description,
                        "category": voice.category,
                        "gender": voice.gender,
                        "quality_score": voice.quality_score
                    },
                    "similarity_score": similarity
                })
            
            return {
                "success": True,
                "reference_voice_id": voice_id,
                "similar_voices": results
            }
            
        except Exception as e:
            logger.error(f"Similar voices search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_voice_recommendations(
        self,
        text: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get voice recommendations for given text and preferences.
        
        Args:
            text: Text to synthesize
            preferences: User preferences
            
        Returns:
            Dictionary with voice recommendations
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            recommendations = await self.cloning_engine.get_voice_recommendations(
                text, preferences or {}
            )
            
            # Convert to serializable format
            results = []
            for voice, score in recommendations:
                results.append({
                    "voice": {
                        "id": voice.id,
                        "name": voice.name,
                        "description": voice.description,
                        "category": voice.category,
                        "gender": voice.gender,
                        "language": voice.language,
                        "quality_score": voice.quality_score
                    },
                    "recommendation_score": score
                })
            
            return {
                "success": True,
                "text_length": len(text),
                "recommendations": results
            }
            
        except Exception as e:
            logger.error(f"Voice recommendations failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_voice_blend(
        self,
        voice_ids: List[str],
        weights: List[float],
        blend_name: str
    ) -> Dict[str, Any]:
        """Create a blended voice from multiple source voices.
        
        Args:
            voice_ids: List of source voice IDs
            weights: List of weights for each voice (must sum to 1.0)
            blend_name: Name for the blended voice
            
        Returns:
            Dictionary with blend result
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            blend_id = await self.cloning_engine.create_voice_blend(
                voice_ids, weights, blend_name
            )
            
            return {
                "success": True,
                "blend_id": blend_id,
                "blend_name": blend_name,
                "source_voices": voice_ids,
                "weights": weights
            }
            
        except Exception as e:
            logger.error(f"Voice blending failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_library_statistics(self) -> Dict[str, Any]:
        """Get voice library statistics.
        
        Returns:
            Dictionary with library statistics
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            stats = await self.cloning_engine.voice_library.get_library_stats()
            
            return {
                "success": True,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Library statistics failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
