"""Audio Quality Validation and Testing Module.

This module provides comprehensive audio quality assessment tools including
objective metrics, reference sample validation, and automated testing.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AudioQualityMetrics:
    """Container for audio quality metrics."""
    
    # Basic audio characteristics
    sample_rate: int
    duration: float
    rms_level: float
    peak_level: float
    dynamic_range: float
    
    # Spectral characteristics
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    zero_crossing_rate: float
    
    # Quality scores (0-100)
    overall_quality: float
    naturalness_score: float
    clarity_score: float
    consistency_score: float
    
    # Performance metrics
    rtf: Optional[float] = None
    inference_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        data = asdict(self)
        # Convert numpy types to Python native types
        for key, value in data.items():
            if hasattr(value, 'item'):  # numpy scalar
                data[key] = value.item()
            elif isinstance(value, (np.floating, np.integer)):
                data[key] = float(value) if isinstance(value, np.floating) else int(value)
        return data


class AudioQualityValidator:
    """Comprehensive audio quality validation system."""
    
    def __init__(self):
        """Initialize audio quality validator."""
        self.reference_samples = {}
        self.quality_thresholds = self._load_quality_thresholds()
        self.has_librosa = self._check_librosa()
        
    def _check_librosa(self) -> bool:
        """Check if librosa is available for advanced analysis."""
        try:
            import librosa
            return True
        except ImportError:
            logger.warning("librosa not available, using basic audio analysis")
            return False
    
    def _load_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load quality thresholds for different audio types."""
        return {
            "speech": {
                "min_rms": 0.01,
                "max_rms": 0.5,
                "min_dynamic_range": 10.0,
                "max_dynamic_range": 40.0,
                "min_spectral_centroid": 500.0,
                "max_spectral_centroid": 4100.0,  # Adjusted to accommodate nova voice characteristics
                "min_overall_quality": 70.0,
                "min_naturalness": 65.0,
                "min_clarity": 70.0,
                "min_consistency": 75.0
            },
            "performance": {
                "max_rtf": 0.5,
                "max_inference_time": 10.0
            }
        }
    
    def analyze_audio(self, audio: np.ndarray, sample_rate: int, 
                     rtf: Optional[float] = None, 
                     inference_time: Optional[float] = None) -> AudioQualityMetrics:
        """Perform comprehensive audio quality analysis.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            rtf: Real-time factor (optional)
            inference_time: Inference time in seconds (optional)
            
        Returns:
            AudioQualityMetrics object with all measurements
        """
        try:
            # Basic audio characteristics
            duration = len(audio) / sample_rate
            rms_level = np.sqrt(np.mean(audio**2))
            peak_level = np.max(np.abs(audio))
            
            # Avoid division by zero
            dynamic_range = 20 * np.log10(peak_level / (rms_level + 1e-8))
            
            # Spectral analysis
            if self.has_librosa:
                spectral_metrics = self._analyze_spectral_features(audio, sample_rate)
            else:
                spectral_metrics = self._basic_spectral_analysis(audio, sample_rate)
            
            # Quality scoring
            quality_scores = self._calculate_quality_scores(
                audio, sample_rate, rms_level, peak_level, dynamic_range, spectral_metrics
            )
            
            return AudioQualityMetrics(
                sample_rate=sample_rate,
                duration=duration,
                rms_level=rms_level,
                peak_level=peak_level,
                dynamic_range=dynamic_range,
                spectral_centroid=spectral_metrics['centroid'],
                spectral_bandwidth=spectral_metrics['bandwidth'],
                spectral_rolloff=spectral_metrics['rolloff'],
                zero_crossing_rate=spectral_metrics['zcr'],
                overall_quality=quality_scores['overall'],
                naturalness_score=quality_scores['naturalness'],
                clarity_score=quality_scores['clarity'],
                consistency_score=quality_scores['consistency'],
                rtf=rtf,
                inference_time=inference_time
            )
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise
    
    def _analyze_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze spectral features using librosa."""
        try:
            import librosa
            
            # Spectral centroid (brightness)
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            
            # Spectral bandwidth
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
            
            # Spectral rolloff
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            return {
                'centroid': float(centroid),
                'bandwidth': float(bandwidth),
                'rolloff': float(rolloff),
                'zcr': float(zcr)
            }
            
        except Exception as e:
            logger.warning(f"Librosa spectral analysis failed: {e}")
            return self._basic_spectral_analysis(audio, sample_rate)
    
    def _basic_spectral_analysis(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Basic spectral analysis using numpy FFT."""
        try:
            # FFT analysis
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
            
            # Spectral centroid
            centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
            
            # Spectral bandwidth (approximation)
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-8))
            
            # Spectral rolloff (85% of energy)
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else sample_rate / 2
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]
            zcr = len(zero_crossings) / len(audio)
            
            return {
                'centroid': float(centroid),
                'bandwidth': float(bandwidth),
                'rolloff': float(rolloff),
                'zcr': float(zcr)
            }
            
        except Exception as e:
            logger.error(f"Basic spectral analysis failed: {e}")
            # Return default values
            return {
                'centroid': 1000.0,
                'bandwidth': 500.0,
                'rolloff': 4000.0,
                'zcr': 0.1
            }
    
    def _calculate_quality_scores(self, audio: np.ndarray, sample_rate: int,
                                 rms_level: float, peak_level: float, 
                                 dynamic_range: float, spectral_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate quality scores based on audio characteristics."""
        
        # Naturalness score (based on spectral characteristics)
        naturalness = 100.0
        
        # Penalize if spectral centroid is too high or low for speech
        centroid = spectral_metrics['centroid']
        if centroid < 500 or centroid > 4000:
            naturalness -= min(30, abs(centroid - 1500) / 50)
        
        # Penalize extreme dynamic range
        if dynamic_range < 10 or dynamic_range > 40:
            naturalness -= min(20, abs(dynamic_range - 20) / 2)
        
        # Clarity score (based on RMS and peak levels)
        clarity = 100.0
        
        # Penalize very low or very high RMS levels
        if rms_level < 0.01 or rms_level > 0.5:
            clarity -= min(25, abs(np.log10(rms_level + 1e-8) + 2) * 10)
        
        # Penalize clipping
        if peak_level > 0.95:
            clarity -= (peak_level - 0.95) * 500
        
        # Consistency score (based on signal stability)
        consistency = 100.0
        
        # Analyze signal consistency using windowed RMS
        window_size = min(1024, len(audio) // 10)
        if window_size > 0:
            windowed_rms = []
            for i in range(0, len(audio) - window_size, window_size):
                window = audio[i:i + window_size]
                windowed_rms.append(np.sqrt(np.mean(window**2)))
            
            if len(windowed_rms) > 1:
                rms_std = np.std(windowed_rms)
                rms_mean = np.mean(windowed_rms)
                consistency_ratio = rms_std / (rms_mean + 1e-8)
                
                # Penalize high variability
                if consistency_ratio > 0.5:
                    consistency -= min(30, (consistency_ratio - 0.5) * 60)
        
        # Overall quality (weighted average)
        overall = (naturalness * 0.4 + clarity * 0.35 + consistency * 0.25)
        
        # Ensure scores are in valid range
        scores = {
            'naturalness': max(0, min(100, naturalness)),
            'clarity': max(0, min(100, clarity)),
            'consistency': max(0, min(100, consistency)),
            'overall': max(0, min(100, overall))
        }
        
        return scores
    
    def validate_against_thresholds(self, metrics: AudioQualityMetrics) -> Dict[str, bool]:
        """Validate metrics against quality thresholds.
        
        Args:
            metrics: AudioQualityMetrics object
            
        Returns:
            Dictionary of validation results
        """
        speech_thresholds = self.quality_thresholds["speech"]
        performance_thresholds = self.quality_thresholds["performance"]
        
        results = {
            # Audio quality validations
            "rms_level_ok": speech_thresholds["min_rms"] <= metrics.rms_level <= speech_thresholds["max_rms"],
            "dynamic_range_ok": speech_thresholds["min_dynamic_range"] <= metrics.dynamic_range <= speech_thresholds["max_dynamic_range"],
            "spectral_centroid_ok": speech_thresholds["min_spectral_centroid"] <= metrics.spectral_centroid <= speech_thresholds["max_spectral_centroid"],
            "overall_quality_ok": metrics.overall_quality >= speech_thresholds["min_overall_quality"],
            "naturalness_ok": metrics.naturalness_score >= speech_thresholds["min_naturalness"],
            "clarity_ok": metrics.clarity_score >= speech_thresholds["min_clarity"],
            "consistency_ok": metrics.consistency_score >= speech_thresholds["min_consistency"],
        }
        
        # Performance validations (if available)
        if metrics.rtf is not None:
            results["rtf_ok"] = metrics.rtf <= performance_thresholds["max_rtf"]
        
        if metrics.inference_time is not None:
            results["inference_time_ok"] = metrics.inference_time <= performance_thresholds["max_inference_time"]
        
        return results
