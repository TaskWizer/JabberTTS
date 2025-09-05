"""JabberTTS Audio Processing Package.

This package contains audio processing functionality including
format conversion, encoding, and streaming capabilities.
"""

from .processor import AudioProcessor, get_audio_processor

__all__ = ["AudioProcessor", "get_audio_processor"]
