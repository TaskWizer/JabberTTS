"""Quality Assessment Framework for TTS Validation.

This module provides comprehensive quality assessment including pronunciation
accuracy, prosody validation, emotion detection, and naturalness scoring.
"""

import logging
import re
import statistics
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class QualityAssessor:
    """Comprehensive TTS quality assessment framework."""
    
    def __init__(self):
        """Initialize quality assessor."""
        self.pronunciation_patterns = self._load_pronunciation_patterns()
        self.prosody_indicators = self._load_prosody_indicators()
        self.emotion_keywords = self._load_emotion_keywords()
        self.naturalness_features = self._load_naturalness_features()
    
    def assess_quality(self, 
                      original_text: str,
                      transcribed_text: str,
                      audio_data: bytes,
                      validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality assessment.
        
        Args:
            original_text: Original input text
            transcribed_text: Whisper transcription
            audio_data: Generated audio data
            validation_result: Whisper validation results
            
        Returns:
            Comprehensive quality assessment results
        """
        try:
            # Extract accuracy metrics from validation result
            accuracy_metrics = validation_result.get("accuracy_metrics", {})
            segment_analysis = validation_result.get("segment_analysis", {})
            
            # Perform individual assessments
            pronunciation_score = self._assess_pronunciation(original_text, transcribed_text, accuracy_metrics)
            prosody_score = self._assess_prosody(original_text, transcribed_text, segment_analysis)
            emotion_score = self._assess_emotion(original_text, transcribed_text)
            naturalness_score = self._assess_naturalness(transcribed_text, segment_analysis)
            audio_quality_score = self._assess_audio_quality(audio_data, segment_analysis)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(
                pronunciation_score,
                prosody_score,
                emotion_score,
                naturalness_score,
                audio_quality_score
            )
            
            # Generate quality report
            quality_report = self._generate_quality_report(
                original_text,
                transcribed_text,
                pronunciation_score,
                prosody_score,
                emotion_score,
                naturalness_score,
                audio_quality_score,
                overall_score
            )
            
            return {
                "overall_score": overall_score,
                "pronunciation_score": pronunciation_score,
                "prosody_score": prosody_score,
                "emotion_score": emotion_score,
                "naturalness_score": naturalness_score,
                "audio_quality_score": audio_quality_score,
                "quality_report": quality_report,
                "assessment_success": True
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "overall_score": 0.0,
                "assessment_success": False,
                "error": str(e)
            }
    
    def _assess_pronunciation(self, original: str, transcribed: str, accuracy_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess pronunciation accuracy.
        
        Args:
            original: Original text
            transcribed: Transcribed text
            accuracy_metrics: Accuracy metrics from Whisper validation
            
        Returns:
            Pronunciation assessment results
        """
        # Base score from word accuracy
        base_score = accuracy_metrics.get("word_accuracy", 0.0)
        
        # Check for common mispronunciations
        mispronunciation_penalty = self._detect_mispronunciations(original, transcribed)
        
        # Check for missing or added words
        word_error_penalty = self._calculate_word_errors(original, transcribed)
        
        # Check for phonetic accuracy
        phonetic_score = self._assess_phonetic_accuracy(original, transcribed)
        
        # Calculate final pronunciation score
        pronunciation_score = max(0.0, base_score - mispronunciation_penalty - word_error_penalty)
        pronunciation_score = (pronunciation_score + phonetic_score) / 2
        
        return {
            "score": pronunciation_score,
            "base_accuracy": base_score,
            "mispronunciation_penalty": mispronunciation_penalty,
            "word_error_penalty": word_error_penalty,
            "phonetic_score": phonetic_score,
            "grade": self._score_to_grade(pronunciation_score)
        }
    
    def _assess_prosody(self, original: str, transcribed: str, segment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess prosody and rhythm quality.
        
        Args:
            original: Original text
            transcribed: Transcribed text
            segment_analysis: Segment analysis from Whisper
            
        Returns:
            Prosody assessment results
        """
        # Analyze speech rate
        speech_rate = segment_analysis.get("speech_rate", 0.0)
        speech_rate_score = self._evaluate_speech_rate(speech_rate)
        
        # Analyze pause patterns
        silence_ratio = segment_analysis.get("silence_ratio", 0.0)
        pause_score = self._evaluate_pause_patterns(silence_ratio)
        
        # Check for proper sentence boundaries
        sentence_boundary_score = self._evaluate_sentence_boundaries(original, segment_analysis)
        
        # Assess rhythm consistency
        rhythm_score = self._assess_rhythm_consistency(segment_analysis)
        
        # Calculate overall prosody score
        prosody_score = (speech_rate_score + pause_score + sentence_boundary_score + rhythm_score) / 4
        
        return {
            "score": prosody_score,
            "speech_rate_score": speech_rate_score,
            "pause_score": pause_score,
            "sentence_boundary_score": sentence_boundary_score,
            "rhythm_score": rhythm_score,
            "speech_rate": speech_rate,
            "silence_ratio": silence_ratio,
            "grade": self._score_to_grade(prosody_score)
        }
    
    def _assess_emotion(self, original: str, transcribed: str) -> Dict[str, Any]:
        """Assess emotional tone preservation.
        
        Args:
            original: Original text
            transcribed: Transcribed text
            
        Returns:
            Emotion assessment results
        """
        # Detect intended emotion from original text
        intended_emotion = self._detect_emotion(original)
        
        # Detect perceived emotion from transcription
        perceived_emotion = self._detect_emotion(transcribed)
        
        # Calculate emotion preservation score
        emotion_match_score = self._calculate_emotion_match(intended_emotion, perceived_emotion)
        
        # Check for emotional markers preservation
        marker_preservation_score = self._assess_emotional_markers(original, transcribed)
        
        # Calculate overall emotion score
        emotion_score = (emotion_match_score + marker_preservation_score) / 2
        
        return {
            "score": emotion_score,
            "intended_emotion": intended_emotion,
            "perceived_emotion": perceived_emotion,
            "emotion_match_score": emotion_match_score,
            "marker_preservation_score": marker_preservation_score,
            "grade": self._score_to_grade(emotion_score)
        }
    
    def _assess_naturalness(self, transcribed: str, segment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess naturalness and human-like characteristics.
        
        Args:
            transcribed: Transcribed text
            segment_analysis: Segment analysis from Whisper
            
        Returns:
            Naturalness assessment results
        """
        # Check transcription confidence (higher confidence suggests clearer speech)
        confidence_score = segment_analysis.get("avg_confidence", 0.0)
        
        # Check speech probability (lower no-speech probability suggests natural speech)
        speech_prob_score = segment_analysis.get("avg_speech_probability", 0.0)
        
        # Assess text completeness and coherence
        coherence_score = self._assess_text_coherence(transcribed)
        
        # Check for robotic patterns
        robotic_penalty = self._detect_robotic_patterns(transcribed, segment_analysis)
        
        # Calculate naturalness score
        naturalness_score = (confidence_score + speech_prob_score + coherence_score) / 3
        naturalness_score = max(0.0, naturalness_score - robotic_penalty)
        
        return {
            "score": naturalness_score,
            "confidence_score": confidence_score,
            "speech_probability_score": speech_prob_score,
            "coherence_score": coherence_score,
            "robotic_penalty": robotic_penalty,
            "grade": self._score_to_grade(naturalness_score)
        }
    
    def _assess_audio_quality(self, audio_data: bytes, segment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess audio quality metrics.
        
        Args:
            audio_data: Generated audio data
            segment_analysis: Segment analysis from Whisper
            
        Returns:
            Audio quality assessment results
        """
        # Use segment analysis as proxy for audio quality
        # High confidence and speech probability suggest good audio quality
        
        confidence_indicator = segment_analysis.get("avg_confidence", 0.0)
        speech_clarity = segment_analysis.get("avg_speech_probability", 0.0)
        
        # Check for audio artifacts (indicated by low confidence or high no-speech probability)
        artifact_penalty = 0.0
        if confidence_indicator < 0.5:
            artifact_penalty += 0.2
        if speech_clarity < 0.7:
            artifact_penalty += 0.2
        
        # Calculate audio quality score
        audio_quality_score = max(0.0, (confidence_indicator + speech_clarity) / 2 - artifact_penalty)
        
        return {
            "score": audio_quality_score,
            "confidence_indicator": confidence_indicator,
            "speech_clarity": speech_clarity,
            "artifact_penalty": artifact_penalty,
            "grade": self._score_to_grade(audio_quality_score)
        }
    
    def _calculate_overall_score(self, pronunciation: Dict, prosody: Dict, emotion: Dict, 
                               naturalness: Dict, audio_quality: Dict) -> Dict[str, Any]:
        """Calculate weighted overall quality score.
        
        Args:
            pronunciation: Pronunciation assessment results
            prosody: Prosody assessment results
            emotion: Emotion assessment results
            naturalness: Naturalness assessment results
            audio_quality: Audio quality assessment results
            
        Returns:
            Overall quality score and breakdown
        """
        # Weighted scoring (pronunciation and naturalness are most important)
        weights = {
            "pronunciation": 0.3,
            "naturalness": 0.25,
            "audio_quality": 0.2,
            "prosody": 0.15,
            "emotion": 0.1
        }
        
        scores = {
            "pronunciation": pronunciation.get("score", 0.0),
            "prosody": prosody.get("score", 0.0),
            "emotion": emotion.get("score", 0.0),
            "naturalness": naturalness.get("score", 0.0),
            "audio_quality": audio_quality.get("score", 0.0)
        }
        
        # Calculate weighted average
        overall_score = sum(scores[key] * weights[key] for key in weights.keys())
        
        return {
            "score": overall_score,
            "weighted_scores": {key: scores[key] * weights[key] for key in weights.keys()},
            "weights": weights,
            "grade": self._score_to_grade(overall_score),
            "quality_level": self._score_to_quality_level(overall_score)
        }
    
    def _generate_quality_report(self, original: str, transcribed: str, pronunciation: Dict,
                               prosody: Dict, emotion: Dict, naturalness: Dict, 
                               audio_quality: Dict, overall: Dict) -> Dict[str, Any]:
        """Generate comprehensive quality report.
        
        Args:
            original: Original text
            transcribed: Transcribed text
            pronunciation: Pronunciation assessment
            prosody: Prosody assessment
            emotion: Emotion assessment
            naturalness: Naturalness assessment
            audio_quality: Audio quality assessment
            overall: Overall assessment
            
        Returns:
            Comprehensive quality report
        """
        # Identify strengths and weaknesses
        scores = {
            "Pronunciation": pronunciation.get("score", 0.0),
            "Prosody": prosody.get("score", 0.0),
            "Emotion": emotion.get("score", 0.0),
            "Naturalness": naturalness.get("score", 0.0),
            "Audio Quality": audio_quality.get("score", 0.0)
        }
        
        strengths = [area for area, score in scores.items() if score >= 0.8]
        weaknesses = [area for area, score in scores.items() if score < 0.6]
        
        # Generate recommendations
        recommendations = []
        if pronunciation.get("score", 0.0) < 0.7:
            recommendations.append("Improve pronunciation accuracy and word clarity")
        if prosody.get("score", 0.0) < 0.7:
            recommendations.append("Enhance speech rhythm and pause patterns")
        if naturalness.get("score", 0.0) < 0.7:
            recommendations.append("Reduce robotic characteristics and improve naturalness")
        if audio_quality.get("score", 0.0) < 0.7:
            recommendations.append("Address audio artifacts and improve signal quality")
        
        return {
            "overall_grade": overall.get("grade", "F"),
            "overall_score": overall.get("score", 0.0),
            "quality_level": overall.get("quality_level", "Poor"),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "detailed_scores": scores,
            "text_comparison": {
                "original_length": len(original.split()),
                "transcribed_length": len(transcribed.split()),
                "length_difference": abs(len(original.split()) - len(transcribed.split()))
            }
        }
    
    # Helper methods for pattern loading and scoring
    def _load_pronunciation_patterns(self) -> Dict[str, List[str]]:
        """Load common pronunciation patterns and errors."""
        return {
            "common_errors": ["th->f", "th->d", "r->w", "l->w"],
            "difficult_words": ["pronunciation", "particularly", "specifically", "literally"]
        }
    
    def _load_prosody_indicators(self) -> Dict[str, Any]:
        """Load prosody analysis indicators."""
        return {
            "optimal_speech_rate": (2.0, 4.0),  # segments per second
            "optimal_silence_ratio": (0.1, 0.3)  # 10-30% silence
        }
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Load emotion detection keywords."""
        return {
            "positive": ["happy", "excited", "wonderful", "amazing", "great"],
            "negative": ["sad", "angry", "terrible", "awful", "horrible"],
            "neutral": ["okay", "fine", "normal", "standard", "regular"]
        }
    
    def _load_naturalness_features(self) -> Dict[str, Any]:
        """Load naturalness assessment features."""
        return {
            "robotic_indicators": ["very", "uniform", "timing", "mechanical"],
            "natural_indicators": ["varied", "flowing", "smooth", "expressive"]
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _score_to_quality_level(self, score: float) -> str:
        """Convert numerical score to quality level."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    # Placeholder implementations for assessment methods
    def _detect_mispronunciations(self, original: str, transcribed: str) -> float:
        """Detect mispronunciations and return penalty score."""
        # Simple implementation - can be enhanced with phonetic analysis
        return 0.0
    
    def _calculate_word_errors(self, original: str, transcribed: str) -> float:
        """Calculate word error penalty."""
        orig_words = set(original.lower().split())
        trans_words = set(transcribed.lower().split())
        missing_words = len(orig_words - trans_words)
        extra_words = len(trans_words - orig_words)
        return (missing_words + extra_words) * 0.05
    
    def _assess_phonetic_accuracy(self, original: str, transcribed: str) -> float:
        """Assess phonetic accuracy."""
        # Simplified implementation
        return 0.8
    
    def _evaluate_speech_rate(self, speech_rate: float) -> float:
        """Evaluate speech rate quality."""
        optimal_range = self.prosody_indicators["optimal_speech_rate"]
        if optimal_range[0] <= speech_rate <= optimal_range[1]:
            return 1.0
        else:
            deviation = min(abs(speech_rate - optimal_range[0]), abs(speech_rate - optimal_range[1]))
            return max(0.0, 1.0 - deviation * 0.2)
    
    def _evaluate_pause_patterns(self, silence_ratio: float) -> float:
        """Evaluate pause pattern quality."""
        optimal_range = self.prosody_indicators["optimal_silence_ratio"]
        if optimal_range[0] <= silence_ratio <= optimal_range[1]:
            return 1.0
        else:
            deviation = min(abs(silence_ratio - optimal_range[0]), abs(silence_ratio - optimal_range[1]))
            return max(0.0, 1.0 - deviation * 2.0)
    
    def _evaluate_sentence_boundaries(self, original: str, segment_analysis: Dict) -> float:
        """Evaluate sentence boundary detection."""
        # Simplified implementation
        return 0.8
    
    def _assess_rhythm_consistency(self, segment_analysis: Dict) -> float:
        """Assess rhythm consistency."""
        # Simplified implementation
        return 0.8
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from text."""
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        return "neutral"
    
    def _calculate_emotion_match(self, intended: str, perceived: str) -> float:
        """Calculate emotion match score."""
        return 1.0 if intended == perceived else 0.5
    
    def _assess_emotional_markers(self, original: str, transcribed: str) -> float:
        """Assess preservation of emotional markers."""
        # Simplified implementation
        return 0.8
    
    def _assess_text_coherence(self, transcribed: str) -> float:
        """Assess text coherence and completeness."""
        # Simple heuristic based on text length and structure
        if len(transcribed.strip()) == 0:
            return 0.0
        return min(1.0, len(transcribed.split()) / 10)  # Normalize by expected length
    
    def _detect_robotic_patterns(self, transcribed: str, segment_analysis: Dict) -> float:
        """Detect robotic speech patterns."""
        # Check for very uniform timing (robotic indicator)
        confidence_variance = 0.1  # Placeholder
        if confidence_variance < 0.05:  # Very uniform = robotic
            return 0.2
        return 0.0
