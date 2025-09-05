"""Whisper-based TTS Validation Module.

This module uses faster-whisper to transcribe TTS output and validate
quality through speech-to-text comparison with original input text.
"""

import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

import soundfile as sf
from jabbertts.config import get_settings

logger = logging.getLogger(__name__)


class WhisperValidator:
    """Whisper-based TTS output validator."""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """Initialize Whisper validator.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run inference on (cpu, cuda)
            compute_type: Compute type for inference (int8, float16, float32)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        self.settings = get_settings()
        
        if WhisperModel is None:
            logger.error("faster-whisper not available. Install with: pip install faster-whisper")
            raise ImportError("faster-whisper is required for validation")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            start_time = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root="./models/whisper"
            )
            
            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio data using Whisper.
        
        Args:
            audio_data: Raw audio data as bytes
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            start_time = time.time()
            
            # Convert audio bytes to numpy array
            with io.BytesIO(audio_data) as audio_buffer:
                audio_array, original_sr = sf.read(audio_buffer)
            
            # Ensure mono audio
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Convert to float32 (required by Whisper)
            audio_array = audio_array.astype(np.float32)

            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if original_sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=16000, dtype=np.float32)
            
            # Transcribe using Whisper
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect transcription results
            transcription_text = ""
            segment_details = []
            
            for segment in segments:
                transcription_text += segment.text
                segment_details.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                })
            
            transcription_time = time.time() - start_time
            
            return {
                "transcription": transcription_text.strip(),
                "segments": segment_details,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "transcription_time": transcription_time,
                "model_info": {
                    "model_size": self.model_size,
                    "device": self.device,
                    "compute_type": self.compute_type
                }
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "transcription": "",
                "segments": [],
                "error": str(e),
                "transcription_time": 0
            }
    
    def validate_tts_output(self, 
                           original_text: str, 
                           audio_data: bytes, 
                           sample_rate: int = 16000) -> Dict[str, Any]:
        """Validate TTS output by comparing original text with transcription.
        
        Args:
            original_text: Original input text
            audio_data: Generated TTS audio data
            sample_rate: Audio sample rate
            
        Returns:
            Validation results with accuracy metrics
        """
        try:
            # Transcribe the audio
            transcription_result = self.transcribe_audio(audio_data, sample_rate)
            
            if "error" in transcription_result:
                return {
                    "success": False,
                    "error": transcription_result["error"],
                    "accuracy_score": 0.0
                }
            
            transcribed_text = transcription_result["transcription"]
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy(original_text, transcribed_text)
            
            # Analyze segments for quality assessment
            segment_analysis = self._analyze_segments(transcription_result["segments"])
            
            return {
                "success": True,
                "original_text": original_text,
                "transcribed_text": transcribed_text,
                "accuracy_metrics": accuracy_metrics,
                "segment_analysis": segment_analysis,
                "transcription_info": {
                    "language": transcription_result.get("language", "unknown"),
                    "language_probability": transcription_result.get("language_probability", 0.0),
                    "duration": transcription_result.get("duration", 0.0),
                    "transcription_time": transcription_result.get("transcription_time", 0.0)
                },
                "model_info": transcription_result.get("model_info", {})
            }
            
        except Exception as e:
            logger.error(f"TTS validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "accuracy_score": 0.0
            }
    
    def _calculate_accuracy(self, original: str, transcribed: str) -> Dict[str, float]:
        """Calculate various accuracy metrics between original and transcribed text.
        
        Args:
            original: Original text
            transcribed: Transcribed text
            
        Returns:
            Dictionary with accuracy metrics
        """
        import difflib
        from textdistance import levenshtein, jaro_winkler, cosine
        
        # Normalize texts for comparison
        original_norm = original.lower().strip()
        transcribed_norm = transcribed.lower().strip()
        
        # Word-level comparison
        original_words = original_norm.split()
        transcribed_words = transcribed_norm.split()
        
        # Calculate various similarity metrics
        metrics = {}
        
        # Character-level Levenshtein distance
        char_distance = levenshtein.distance(original_norm, transcribed_norm)
        char_similarity = 1.0 - (char_distance / max(len(original_norm), len(transcribed_norm), 1))
        metrics["character_accuracy"] = max(0.0, char_similarity)
        
        # Word-level accuracy
        word_distance = levenshtein.distance(original_words, transcribed_words)
        word_similarity = 1.0 - (word_distance / max(len(original_words), len(transcribed_words), 1))
        metrics["word_accuracy"] = max(0.0, word_similarity)
        
        # Jaro-Winkler similarity
        metrics["jaro_winkler_similarity"] = jaro_winkler.similarity(original_norm, transcribed_norm)
        
        # Cosine similarity (character n-grams)
        metrics["cosine_similarity"] = cosine.similarity(original_norm, transcribed_norm)
        
        # Sequence matcher ratio
        sequence_matcher = difflib.SequenceMatcher(None, original_norm, transcribed_norm)
        metrics["sequence_match_ratio"] = sequence_matcher.ratio()
        
        # Overall accuracy score (weighted average)
        metrics["overall_accuracy"] = (
            metrics["word_accuracy"] * 0.4 +
            metrics["character_accuracy"] * 0.3 +
            metrics["jaro_winkler_similarity"] * 0.2 +
            metrics["sequence_match_ratio"] * 0.1
        )
        
        return metrics
    
    def _analyze_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transcription segments for quality indicators.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Segment analysis results
        """
        if not segments:
            return {
                "segment_count": 0,
                "avg_confidence": 0.0,
                "speech_rate": 0.0,
                "silence_ratio": 0.0
            }
        
        # Calculate confidence metrics
        confidences = []
        speech_probs = []
        total_duration = 0
        total_speech_duration = 0
        
        for segment in segments:
            if "avg_logprob" in segment:
                # Convert log probability to confidence (approximate)
                confidence = np.exp(segment["avg_logprob"])
                confidences.append(confidence)
            
            if "no_speech_prob" in segment:
                speech_prob = 1.0 - segment["no_speech_prob"]
                speech_probs.append(speech_prob)
            
            # Calculate durations
            duration = segment.get("end", 0) - segment.get("start", 0)
            total_duration += duration
            
            if segment.get("text", "").strip():
                total_speech_duration += duration
        
        # Calculate metrics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_speech_prob = np.mean(speech_probs) if speech_probs else 0.0
        speech_rate = len(segments) / total_duration if total_duration > 0 else 0.0
        silence_ratio = (total_duration - total_speech_duration) / total_duration if total_duration > 0 else 0.0
        
        return {
            "segment_count": len(segments),
            "avg_confidence": float(avg_confidence),
            "avg_speech_probability": float(avg_speech_prob),
            "speech_rate": float(speech_rate),
            "silence_ratio": float(silence_ratio),
            "total_duration": float(total_duration),
            "speech_duration": float(total_speech_duration)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded Whisper model.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_loaded": self.model is not None,
            "available": WhisperModel is not None
        }


# Global validator instance
_whisper_validator: Optional[WhisperValidator] = None


def get_whisper_validator(model_size: str = "base") -> WhisperValidator:
    """Get the global Whisper validator instance.
    
    Args:
        model_size: Whisper model size to use
        
    Returns:
        Global WhisperValidator instance
    """
    global _whisper_validator
    if _whisper_validator is None or _whisper_validator.model_size != model_size:
        _whisper_validator = WhisperValidator(model_size=model_size)
    return _whisper_validator
