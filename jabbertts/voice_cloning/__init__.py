"""Voice Cloning Studio Module.

This module provides comprehensive voice cloning capabilities including:
- Speaker embedding extraction and analysis
- Voice characteristic manipulation and control
- Real-time voice synthesis preview
- Voice library management and organization
- Custom voice model training and fine-tuning
"""

from .embedding_extractor import VoiceEmbeddingExtractor
from .voice_modulator import VoiceModulator
from .voice_library import VoiceLibraryManager
from .cloning_engine import VoiceCloningEngine
from .studio_interface import VoiceCloningStudio

__all__ = [
    "VoiceEmbeddingExtractor",
    "VoiceModulator", 
    "VoiceLibraryManager",
    "VoiceCloningEngine",
    "VoiceCloningStudio"
]
