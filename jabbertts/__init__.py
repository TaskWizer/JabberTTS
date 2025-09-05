"""JabberTTS - Fast, efficient Text-to-Speech API with OpenAI compatibility.

JabberTTS provides a high-performance, CPU-optimized text-to-speech API
that is fully compatible with OpenAI's TTS API. It features voice cloning,
multiple audio formats, and near-instant response times on consumer hardware.

Key Features:
- RTF < 0.5 on CPU-only hardware
- OpenAI-compatible API endpoints
- Voice cloning from audio samples
- Multiple audio format support
- Streaming audio responses
- Memory usage < 2GB

Example:
    Basic usage with the API:
    
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/v1/audio/speech",
        json={
            "model": "openaudio-s1-mini",
            "input": "Hello, world!",
            "voice": "alloy"
        }
    )
    
    with open("output.mp3", "wb") as f:
        f.write(response.content)
    ```
"""

__version__ = "0.1.0"
__author__ = "JabberTTS Team"
__email__ = "contact@jabbertts.com"
__license__ = "Apache-2.0"

# Version info tuple for programmatic access
VERSION_INFO = tuple(int(x) for x in __version__.split("."))

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "VERSION_INFO",
]
