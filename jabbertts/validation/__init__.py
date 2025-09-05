"""JabberTTS Validation Module.

This module provides automated quality assurance and validation capabilities
using Whisper-based speech-to-text transcription for TTS output verification.
"""

from .whisper_validator import WhisperValidator, get_whisper_validator
from .quality_assessor import QualityAssessor
from .test_suite import ValidationTestSuite, get_validation_test_suite
from .metrics import ValidationMetrics, get_validation_metrics
from .self_debugger import SelfDebugger, get_self_debugger
from .audio_quality import AudioQualityValidator, AudioQualityMetrics

__all__ = [
    "WhisperValidator",
    "get_whisper_validator",
    "QualityAssessor",
    "ValidationTestSuite",
    "get_validation_test_suite",
    "ValidationMetrics",
    "get_validation_metrics",
    "SelfDebugger",
    "get_self_debugger",
    "AudioQualityValidator",
    "AudioQualityMetrics"
]
