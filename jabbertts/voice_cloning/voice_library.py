"""Voice Library Manager for Custom Voice Organization.

This module provides comprehensive voice library management including
voice storage, organization, metadata management, and search capabilities.
"""

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import soundfile as sf

from .embedding_extractor import VoiceCharacteristics, VoiceEmbeddingExtractor
from .voice_modulator import ModulationParameters

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Complete voice profile with metadata and characteristics."""
    id: str
    name: str
    description: str
    characteristics: VoiceCharacteristics
    audio_samples: List[str]  # Paths to audio samples
    modulation_presets: List[Dict[str, any]]
    tags: List[str]
    category: str
    language: str
    gender: str
    age_range: str
    quality_score: float
    created_at: str
    updated_at: str
    usage_count: int
    is_favorite: bool
    is_public: bool
    creator: str
    source_type: str  # "uploaded", "cloned", "generated"


class VoiceLibraryManager:
    """Comprehensive voice library management system."""
    
    def __init__(self, library_path: Union[str, Path] = None):
        """Initialize the voice library manager.
        
        Args:
            library_path: Path to voice library directory
        """
        self.library_path = Path(library_path) if library_path else Path("voice_library")
        self.voices_path = self.library_path / "voices"
        self.metadata_path = self.library_path / "metadata"
        self.samples_path = self.library_path / "samples"
        self.presets_path = self.library_path / "presets"
        
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.embedding_extractor = None
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the voice library manager."""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing voice library manager...")
            
            # Create directory structure
            self._create_directory_structure()
            
            # Initialize embedding extractor
            self.embedding_extractor = VoiceEmbeddingExtractor()
            await self.embedding_extractor.initialize()
            
            # Load existing voice profiles
            await self._load_voice_profiles()
            
            self.is_initialized = True
            logger.info(f"Voice library manager initialized with {len(self.voice_profiles)} voices")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice library manager: {e}")
            raise
    
    def _create_directory_structure(self) -> None:
        """Create the voice library directory structure."""
        directories = [
            self.library_path,
            self.voices_path,
            self.metadata_path,
            self.samples_path,
            self.presets_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _load_voice_profiles(self) -> None:
        """Load existing voice profiles from metadata files."""
        try:
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        profile_data = json.load(f)
                    
                    # Reconstruct VoiceCharacteristics
                    char_data = profile_data['characteristics']
                    characteristics = VoiceCharacteristics(
                        embedding=np.array(char_data['embedding']),
                        fundamental_frequency=char_data['fundamental_frequency'],
                        formant_frequencies=char_data['formant_frequencies'],
                        spectral_centroid=char_data['spectral_centroid'],
                        spectral_rolloff=char_data['spectral_rolloff'],
                        zero_crossing_rate=char_data['zero_crossing_rate'],
                        mfcc_features=np.array(char_data['mfcc_features']),
                        pitch_range=tuple(char_data['pitch_range']),
                        voice_quality_metrics=char_data['voice_quality_metrics'],
                        confidence_score=char_data['confidence_score']
                    )
                    
                    # Create VoiceProfile
                    profile = VoiceProfile(
                        id=profile_data['id'],
                        name=profile_data['name'],
                        description=profile_data['description'],
                        characteristics=characteristics,
                        audio_samples=profile_data['audio_samples'],
                        modulation_presets=profile_data['modulation_presets'],
                        tags=profile_data['tags'],
                        category=profile_data['category'],
                        language=profile_data['language'],
                        gender=profile_data['gender'],
                        age_range=profile_data['age_range'],
                        quality_score=profile_data['quality_score'],
                        created_at=profile_data['created_at'],
                        updated_at=profile_data['updated_at'],
                        usage_count=profile_data['usage_count'],
                        is_favorite=profile_data['is_favorite'],
                        is_public=profile_data['is_public'],
                        creator=profile_data['creator'],
                        source_type=profile_data['source_type']
                    )
                    
                    self.voice_profiles[profile.id] = profile
                    
                except Exception as e:
                    logger.warning(f"Failed to load voice profile from {metadata_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")
    
    async def add_voice_from_audio(
        self,
        name: str,
        audio_data: Union[bytes, np.ndarray, str, Path],
        description: str = "",
        tags: List[str] = None,
        category: str = "custom",
        language: str = "en",
        creator: str = "user"
    ) -> str:
        """Add a new voice to the library from audio data.
        
        Args:
            name: Voice name
            audio_data: Audio data (bytes, numpy array, or file path)
            description: Voice description
            tags: Voice tags
            category: Voice category
            language: Voice language
            creator: Voice creator
            
        Returns:
            Voice ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Generate unique voice ID
            voice_id = str(uuid.uuid4())
            
            # Extract voice characteristics
            characteristics = await self.embedding_extractor.extract_voice_characteristics(audio_data)
            
            # Save audio sample
            sample_filename = f"{voice_id}_sample.wav"
            sample_path = self.samples_path / sample_filename
            
            if isinstance(audio_data, (str, Path)):
                # Copy file
                shutil.copy2(str(audio_data), str(sample_path))
            else:
                # Save audio data
                if isinstance(audio_data, bytes):
                    import io
                    audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
                else:
                    audio_array = audio_data
                    sample_rate = 24000  # Default sample rate
                
                sf.write(str(sample_path), audio_array, sample_rate)
            
            # Determine gender and age from characteristics
            gender = self._estimate_gender(characteristics)
            age_range = self._estimate_age_range(characteristics)
            
            # Create voice profile
            profile = VoiceProfile(
                id=voice_id,
                name=name,
                description=description,
                characteristics=characteristics,
                audio_samples=[sample_filename],
                modulation_presets=[],
                tags=tags or [],
                category=category,
                language=language,
                gender=gender,
                age_range=age_range,
                quality_score=characteristics.confidence_score,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                usage_count=0,
                is_favorite=False,
                is_public=False,
                creator=creator,
                source_type="uploaded"
            )
            
            # Save profile
            await self._save_voice_profile(profile)
            
            # Add to memory
            self.voice_profiles[voice_id] = profile
            
            logger.info(f"Added voice '{name}' with ID {voice_id}")
            return voice_id
            
        except Exception as e:
            logger.error(f"Failed to add voice from audio: {e}")
            raise
    
    def _estimate_gender(self, characteristics: VoiceCharacteristics) -> str:
        """Estimate gender from voice characteristics."""
        try:
            # Simple gender estimation based on fundamental frequency
            f0 = characteristics.fundamental_frequency
            
            if f0 < 120:
                return "male"
            elif f0 > 200:
                return "female"
            else:
                return "neutral"
                
        except Exception:
            return "unknown"
    
    def _estimate_age_range(self, characteristics: VoiceCharacteristics) -> str:
        """Estimate age range from voice characteristics."""
        try:
            # Simple age estimation based on voice quality metrics
            jitter = characteristics.voice_quality_metrics.get('jitter', 0.0)
            shimmer = characteristics.voice_quality_metrics.get('shimmer', 0.0)
            
            if jitter < 0.01 and shimmer < 0.05:
                return "young"
            elif jitter < 0.02 and shimmer < 0.1:
                return "adult"
            else:
                return "mature"
                
        except Exception:
            return "unknown"
    
    async def _save_voice_profile(self, profile: VoiceProfile) -> None:
        """Save voice profile to metadata file."""
        try:
            metadata_file = self.metadata_path / f"{profile.id}.json"
            
            # Convert profile to dictionary
            profile_dict = asdict(profile)
            
            # Convert numpy arrays to lists for JSON serialization
            char_dict = profile_dict['characteristics']
            char_dict['embedding'] = char_dict['embedding'].tolist()
            char_dict['mfcc_features'] = char_dict['mfcc_features'].tolist()
            
            with open(metadata_file, 'w') as f:
                json.dump(profile_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save voice profile: {e}")
            raise
    
    async def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile by ID."""
        if not self.is_initialized:
            await self.initialize()
        
        return self.voice_profiles.get(voice_id)
    
    async def list_voices(
        self,
        category: Optional[str] = None,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        tags: Optional[List[str]] = None,
        favorites_only: bool = False
    ) -> List[VoiceProfile]:
        """List voices with optional filtering."""
        if not self.is_initialized:
            await self.initialize()
        
        voices = list(self.voice_profiles.values())
        
        # Apply filters
        if category:
            voices = [v for v in voices if v.category == category]
        
        if language:
            voices = [v for v in voices if v.language == language]
        
        if gender:
            voices = [v for v in voices if v.gender == gender]
        
        if tags:
            voices = [v for v in voices if any(tag in v.tags for tag in tags)]
        
        if favorites_only:
            voices = [v for v in voices if v.is_favorite]
        
        # Sort by usage count and quality
        voices.sort(key=lambda v: (v.usage_count, v.quality_score), reverse=True)
        
        return voices
    
    async def search_voices(self, query: str) -> List[VoiceProfile]:
        """Search voices by name, description, or tags."""
        if not self.is_initialized:
            await self.initialize()
        
        query_lower = query.lower()
        matching_voices = []
        
        for voice in self.voice_profiles.values():
            if (query_lower in voice.name.lower() or
                query_lower in voice.description.lower() or
                any(query_lower in tag.lower() for tag in voice.tags)):
                matching_voices.append(voice)
        
        return matching_voices
    
    async def find_similar_voices(
        self, 
        voice_id: str, 
        max_results: int = 5
    ) -> List[Tuple[VoiceProfile, float]]:
        """Find voices similar to the specified voice."""
        if not self.is_initialized:
            await self.initialize()
        
        target_voice = self.voice_profiles.get(voice_id)
        if not target_voice:
            return []
        
        similarities = []
        
        for other_voice in self.voice_profiles.values():
            if other_voice.id == voice_id:
                continue
            
            # Calculate similarity
            similarity_metrics = await self.embedding_extractor.compare_voices(
                target_voice.characteristics,
                other_voice.characteristics
            )
            
            overall_similarity = similarity_metrics.get('overall_similarity', 0.0)
            similarities.append((other_voice, overall_similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    async def update_voice_metadata(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        is_favorite: Optional[bool] = None
    ) -> bool:
        """Update voice metadata."""
        if not self.is_initialized:
            await self.initialize()
        
        voice = self.voice_profiles.get(voice_id)
        if not voice:
            return False
        
        try:
            # Update fields
            if name is not None:
                voice.name = name
            if description is not None:
                voice.description = description
            if tags is not None:
                voice.tags = tags
            if category is not None:
                voice.category = category
            if is_favorite is not None:
                voice.is_favorite = is_favorite
            
            voice.updated_at = datetime.now().isoformat()
            
            # Save updated profile
            await self._save_voice_profile(voice)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update voice metadata: {e}")
            return False
    
    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice from the library."""
        if not self.is_initialized:
            await self.initialize()
        
        voice = self.voice_profiles.get(voice_id)
        if not voice:
            return False
        
        try:
            # Delete audio samples
            for sample_filename in voice.audio_samples:
                sample_path = self.samples_path / sample_filename
                if sample_path.exists():
                    sample_path.unlink()
            
            # Delete metadata file
            metadata_file = self.metadata_path / f"{voice_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from memory
            del self.voice_profiles[voice_id]
            
            logger.info(f"Deleted voice '{voice.name}' with ID {voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice: {e}")
            return False
    
    async def increment_usage_count(self, voice_id: str) -> None:
        """Increment usage count for a voice."""
        voice = self.voice_profiles.get(voice_id)
        if voice:
            voice.usage_count += 1
            voice.updated_at = datetime.now().isoformat()
            await self._save_voice_profile(voice)
    
    async def get_library_stats(self) -> Dict[str, any]:
        """Get voice library statistics."""
        if not self.is_initialized:
            await self.initialize()
        
        voices = list(self.voice_profiles.values())
        
        stats = {
            "total_voices": len(voices),
            "categories": {},
            "languages": {},
            "genders": {},
            "source_types": {},
            "favorites_count": sum(1 for v in voices if v.is_favorite),
            "average_quality": np.mean([v.quality_score for v in voices]) if voices else 0.0,
            "total_usage": sum(v.usage_count for v in voices)
        }
        
        # Count by categories
        for voice in voices:
            stats["categories"][voice.category] = stats["categories"].get(voice.category, 0) + 1
            stats["languages"][voice.language] = stats["languages"].get(voice.language, 0) + 1
            stats["genders"][voice.gender] = stats["genders"].get(voice.gender, 0) + 1
            stats["source_types"][voice.source_type] = stats["source_types"].get(voice.source_type, 0) + 1
        
        return stats
