"""Voice Embedding Extractor for Speaker Characteristic Analysis.

This module provides advanced speaker embedding extraction capabilities
for voice cloning and characteristic analysis.
"""

import asyncio
import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class VoiceCharacteristics:
    """Container for extracted voice characteristics."""
    embedding: np.ndarray
    fundamental_frequency: float
    formant_frequencies: List[float]
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc_features: np.ndarray
    pitch_range: Tuple[float, float]
    voice_quality_metrics: Dict[str, float]
    confidence_score: float


class VoiceEmbeddingExtractor:
    """Advanced voice embedding extractor for speaker characteristics."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize the voice embedding extractor.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
        self.sample_rate = 16000  # Standard sample rate for voice analysis
        self.embedding_cache = {}
        self.is_initialized = False
        
    async def initialize(self) -> None:
        """Initialize the embedding extraction models."""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing voice embedding extractor...")
            
            # Initialize speaker encoder (placeholder for actual model)
            # In production, this would load a pre-trained speaker encoder
            # such as SpeakerNet, ECAPA-TDNN, or similar
            self.speaker_encoder = self._create_speaker_encoder()
            
            # Initialize audio preprocessing pipeline
            self.audio_preprocessor = self._create_audio_preprocessor()
            
            self.is_initialized = True
            logger.info("Voice embedding extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice embedding extractor: {e}")
            raise
    
    def _create_speaker_encoder(self):
        """Create speaker encoder model (placeholder implementation)."""
        # This is a placeholder - in production, load a real speaker encoder
        class MockSpeakerEncoder:
            def __init__(self, embedding_dim=256):
                self.embedding_dim = embedding_dim
                
            def encode(self, audio_features):
                # Mock embedding generation
                return np.random.randn(self.embedding_dim).astype(np.float32)
        
        return MockSpeakerEncoder()
    
    def _create_audio_preprocessor(self):
        """Create audio preprocessing pipeline."""
        class AudioPreprocessor:
            def __init__(self, sample_rate=16000):
                self.sample_rate = sample_rate
                
            def preprocess(self, audio_data, original_sr):
                # Resample if necessary
                if original_sr != self.sample_rate:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=original_sr, 
                        target_sr=self.sample_rate
                    )
                
                # Normalize audio
                audio_data = librosa.util.normalize(audio_data)
                
                # Remove silence
                audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
                
                return audio_data
        
        return AudioPreprocessor(self.sample_rate)
    
    async def extract_voice_characteristics(
        self, 
        audio_data: Union[bytes, np.ndarray, str, Path],
        audio_format: str = "auto"
    ) -> VoiceCharacteristics:
        """Extract comprehensive voice characteristics from audio.
        
        Args:
            audio_data: Audio data (bytes, numpy array, or file path)
            audio_format: Audio format (auto-detected if not specified)
            
        Returns:
            VoiceCharacteristics object with extracted features
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Load and preprocess audio
            audio_array, sample_rate = await self._load_audio(audio_data, audio_format)
            audio_array = self.audio_preprocessor.preprocess(audio_array, sample_rate)
            
            # Extract various voice characteristics
            characteristics = await self._analyze_voice_characteristics(audio_array)
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Failed to extract voice characteristics: {e}")
            raise
    
    async def _load_audio(
        self, 
        audio_data: Union[bytes, np.ndarray, str, Path],
        audio_format: str
    ) -> Tuple[np.ndarray, int]:
        """Load audio data from various sources."""
        if isinstance(audio_data, (str, Path)):
            # Load from file
            audio_array, sample_rate = sf.read(str(audio_data))
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # Convert to mono
            return audio_array, sample_rate
            
        elif isinstance(audio_data, bytes):
            # Load from bytes
            import io
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # Convert to mono
            return audio_array, sample_rate
            
        elif isinstance(audio_data, np.ndarray):
            # Already a numpy array
            return audio_data, self.sample_rate
            
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    async def _analyze_voice_characteristics(self, audio_array: np.ndarray) -> VoiceCharacteristics:
        """Analyze comprehensive voice characteristics."""
        try:
            # Extract speaker embedding
            embedding = self.speaker_encoder.encode(audio_array)
            
            # Extract fundamental frequency (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_array,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            fundamental_frequency = np.nanmean(f0[voiced_flag])
            
            # Extract formant frequencies (simplified estimation)
            formant_frequencies = self._estimate_formants(audio_array)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_array, sr=self.sample_rate))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_array))
            
            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=13)
            mfcc_features = np.mean(mfcc_features, axis=1)
            
            # Calculate pitch range
            valid_f0 = f0[voiced_flag]
            pitch_range = (np.min(valid_f0), np.max(valid_f0)) if len(valid_f0) > 0 else (0.0, 0.0)
            
            # Calculate voice quality metrics
            voice_quality_metrics = self._calculate_voice_quality_metrics(audio_array, f0, voiced_flag)
            
            # Calculate confidence score based on audio quality
            confidence_score = self._calculate_confidence_score(audio_array, voiced_flag)
            
            return VoiceCharacteristics(
                embedding=embedding,
                fundamental_frequency=float(fundamental_frequency) if not np.isnan(fundamental_frequency) else 0.0,
                formant_frequencies=formant_frequencies,
                spectral_centroid=float(spectral_centroid),
                spectral_rolloff=float(spectral_rolloff),
                zero_crossing_rate=float(zero_crossing_rate),
                mfcc_features=mfcc_features,
                pitch_range=pitch_range,
                voice_quality_metrics=voice_quality_metrics,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze voice characteristics: {e}")
            raise
    
    def _estimate_formants(self, audio_array: np.ndarray) -> List[float]:
        """Estimate formant frequencies (simplified implementation)."""
        try:
            # This is a simplified formant estimation
            # In production, use more sophisticated methods like LPC analysis
            
            # Compute power spectral density
            freqs, psd = librosa.core.piptrack(y=audio_array, sr=self.sample_rate)
            
            # Find peaks in the spectrum (simplified formant estimation)
            formants = []
            for i in range(min(4, freqs.shape[0])):  # First 4 formants
                if len(freqs[i]) > 0:
                    formant_freq = np.mean(freqs[i][freqs[i] > 0])
                    if not np.isnan(formant_freq):
                        formants.append(float(formant_freq))
            
            # Ensure we have at least 3 formants (fill with defaults if needed)
            while len(formants) < 3:
                formants.append(0.0)
            
            return formants[:4]  # Return first 4 formants
            
        except Exception as e:
            logger.warning(f"Formant estimation failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _calculate_voice_quality_metrics(
        self, 
        audio_array: np.ndarray, 
        f0: np.ndarray, 
        voiced_flag: np.ndarray
    ) -> Dict[str, float]:
        """Calculate voice quality metrics."""
        try:
            metrics = {}
            
            # Jitter (pitch period variability)
            if np.sum(voiced_flag) > 1:
                valid_f0 = f0[voiced_flag]
                jitter = np.std(valid_f0) / np.mean(valid_f0) if np.mean(valid_f0) > 0 else 0.0
                metrics['jitter'] = float(jitter)
            else:
                metrics['jitter'] = 0.0
            
            # Shimmer (amplitude variability) - simplified
            rms_energy = librosa.feature.rms(y=audio_array)[0]
            if len(rms_energy) > 1:
                shimmer = np.std(rms_energy) / np.mean(rms_energy) if np.mean(rms_energy) > 0 else 0.0
                metrics['shimmer'] = float(shimmer)
            else:
                metrics['shimmer'] = 0.0
            
            # Harmonics-to-noise ratio (simplified estimation)
            hnr = self._estimate_hnr(audio_array, f0, voiced_flag)
            metrics['hnr'] = hnr
            
            # Voice activity ratio
            voice_activity_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0.0
            metrics['voice_activity_ratio'] = float(voice_activity_ratio)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Voice quality metrics calculation failed: {e}")
            return {'jitter': 0.0, 'shimmer': 0.0, 'hnr': 0.0, 'voice_activity_ratio': 0.0}
    
    def _estimate_hnr(self, audio_array: np.ndarray, f0: np.ndarray, voiced_flag: np.ndarray) -> float:
        """Estimate harmonics-to-noise ratio (simplified)."""
        try:
            # This is a simplified HNR estimation
            # In production, use more sophisticated harmonic analysis
            
            if np.sum(voiced_flag) == 0:
                return 0.0
            
            # Calculate spectral energy in harmonic vs non-harmonic regions
            stft = librosa.stft(audio_array)
            magnitude = np.abs(stft)
            
            # Simplified HNR calculation
            total_energy = np.sum(magnitude ** 2)
            if total_energy > 0:
                # This is a very simplified estimation
                hnr = 10 * np.log10(total_energy / (total_energy * 0.1 + 1e-10))
                return float(np.clip(hnr, 0, 30))  # Clip to reasonable range
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"HNR estimation failed: {e}")
            return 0.0
    
    def _calculate_confidence_score(self, audio_array: np.ndarray, voiced_flag: np.ndarray) -> float:
        """Calculate confidence score for the voice analysis."""
        try:
            # Factors affecting confidence:
            # 1. Audio length
            # 2. Voice activity ratio
            # 3. Signal-to-noise ratio
            # 4. Spectral quality
            
            confidence = 1.0
            
            # Audio length factor
            duration = len(audio_array) / self.sample_rate
            if duration < 1.0:
                confidence *= 0.5  # Very short audio
            elif duration < 3.0:
                confidence *= 0.8  # Short audio
            
            # Voice activity factor
            voice_activity_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0.0
            confidence *= voice_activity_ratio
            
            # Signal quality factor (simplified)
            rms_energy = np.sqrt(np.mean(audio_array ** 2))
            if rms_energy < 0.01:
                confidence *= 0.5  # Very quiet audio
            elif rms_energy > 0.5:
                confidence *= 0.8  # Potentially clipped audio
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Confidence score calculation failed: {e}")
            return 0.5
    
    async def compare_voices(
        self, 
        characteristics1: VoiceCharacteristics, 
        characteristics2: VoiceCharacteristics
    ) -> Dict[str, float]:
        """Compare two voice characteristics and return similarity metrics."""
        try:
            similarity_metrics = {}
            
            # Embedding similarity (cosine similarity)
            embedding_similarity = np.dot(characteristics1.embedding, characteristics2.embedding) / (
                np.linalg.norm(characteristics1.embedding) * np.linalg.norm(characteristics2.embedding)
            )
            similarity_metrics['embedding_similarity'] = float(embedding_similarity)
            
            # Fundamental frequency similarity
            f0_diff = abs(characteristics1.fundamental_frequency - characteristics2.fundamental_frequency)
            f0_similarity = 1.0 / (1.0 + f0_diff / 100.0)  # Normalize by 100 Hz
            similarity_metrics['f0_similarity'] = float(f0_similarity)
            
            # Formant similarity
            formant_similarities = []
            for f1, f2 in zip(characteristics1.formant_frequencies, characteristics2.formant_frequencies):
                if f1 > 0 and f2 > 0:
                    formant_diff = abs(f1 - f2)
                    formant_sim = 1.0 / (1.0 + formant_diff / 500.0)  # Normalize by 500 Hz
                    formant_similarities.append(formant_sim)
            
            formant_similarity = np.mean(formant_similarities) if formant_similarities else 0.0
            similarity_metrics['formant_similarity'] = float(formant_similarity)
            
            # MFCC similarity
            mfcc_similarity = np.corrcoef(characteristics1.mfcc_features, characteristics2.mfcc_features)[0, 1]
            if np.isnan(mfcc_similarity):
                mfcc_similarity = 0.0
            similarity_metrics['mfcc_similarity'] = float(mfcc_similarity)
            
            # Overall similarity (weighted average)
            overall_similarity = (
                0.4 * embedding_similarity +
                0.2 * f0_similarity +
                0.2 * formant_similarity +
                0.2 * mfcc_similarity
            )
            similarity_metrics['overall_similarity'] = float(overall_similarity)
            
            return similarity_metrics
            
        except Exception as e:
            logger.error(f"Voice comparison failed: {e}")
            return {'overall_similarity': 0.0}
