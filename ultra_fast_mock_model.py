
import numpy as np
import time
from jabbertts.models.base import BaseTTSModel
from pathlib import Path

class UltraFastMockModel(BaseTTSModel):
    """Ultra-fast mock TTS model for performance testing."""

    DESCRIPTION = "Ultra-fast mock TTS model - optimized for RTF < 0.1"
    SAMPLE_RATE = 22050

    def __init__(self, model_path: Path, device: str = "cpu"):
        super().__init__(model_path, device)
        self.is_loaded = False

    def load_model(self) -> None:
        """Load mock model (instant)."""
        self.is_loaded = True

    def unload_model(self) -> None:
        """Unload mock model."""
        self.is_loaded = False

    def generate_speech(self, text: str, voice: str = "alloy", speed: float = 1.0, **kwargs) -> np.ndarray:
        """Generate mock speech very quickly."""
        # Calculate expected audio duration based on text length
        # Rough estimate: 150 words per minute, 5 chars per word
        chars_per_second = (150 * 5) / 60  # ~12.5 chars/second
        audio_duration = len(text) / chars_per_second

        # Generate audio samples
        num_samples = int(audio_duration * self.SAMPLE_RATE)

        # Create realistic-sounding audio (sine wave with noise)
        t = np.linspace(0, audio_duration, num_samples)
        frequency = 200 + np.random.random() * 300  # Random frequency 200-500 Hz
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)

        # Add some noise for realism
        noise = 0.05 * np.random.random(num_samples) - 0.025
        audio = audio + noise

        # Apply speed adjustment
        if speed != 1.0:
            new_length = int(len(audio) / speed)
            audio = np.interp(np.linspace(0, len(audio), new_length), np.arange(len(audio)), audio)

        return audio.astype(np.float32)

    def get_supported_voices(self):
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def get_sample_rate(self):
        return self.SAMPLE_RATE

    def validate_input(self, text: str) -> bool:
        return len(text) > 0 and len(text) <= 4096
