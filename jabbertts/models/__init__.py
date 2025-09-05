"""JabberTTS Models Package.

This package contains model management, loading, and inference functionality
for various TTS models including OpenAudio S1-mini and alternatives.
"""

from .manager import ModelManager
from .base import BaseTTSModel

__all__ = ["ModelManager", "BaseTTSModel"]
