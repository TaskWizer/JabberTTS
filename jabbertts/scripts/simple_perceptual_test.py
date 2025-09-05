#!/usr/bin/env python3
"""
Simple perceptual quality test without complex dependencies.

This script validates the perceptual quality framework with basic metrics.
"""

import asyncio
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.validation.audio_quality import AudioQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SimplePerceptualMetrics:
    """Simple perceptual quality metrics."""
    transcription_accuracy: float
    word_error_rate: float
    overall_quality: float
    naturalness_score: float
    clarity_score: float
    prosody_score: float
    human_likeness: float
    rtf: float
    voice: str
    timestamp: str


def calculate_simple_prosody(audio_data: np.ndarray, sample_rate: int, text: str) -> float:
    """Calculate simple prosody score."""
    try:
        if len(audio_data) == 0:
            return 0.0
        
        duration = len(audio_data) / sample_rate
        text_length = len(text.split())
        
        # Speech rate analysis
        speech_rate = (text_length / duration) * 60 if duration > 0 else 0
        ideal_rate = 165  # words per minute
        rate_deviation = abs(speech_rate - ideal_rate) / ideal_rate
        rate_score = max(0, 100 - (rate_deviation * 100))
        
        # Amplitude variation analysis
        window_size = int(sample_rate * 0.05)  # 50ms windows
        if window_size > 0 and len(audio_data) > window_size:
            amplitude_envelope = []
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                amplitude_envelope.append(np.sqrt(np.mean(window ** 2)))
            
            if len(amplitude_envelope) > 1:
                amplitude_variation = np.std(amplitude_envelope) / (np.mean(amplitude_envelope) + 1e-8)
                variation_score = min(100, amplitude_variation * 200)
                prosody_score = (rate_score + variation_score) / 2
            else:
                prosody_score = rate_score
        else:
            prosody_score = rate_score
        
        return min(100, prosody_score)
        
    except Exception as e:
        logger.warning(f"Prosody calculation failed: {e}")
        return 50.0


def calculate_human_likeness(naturalness: float, prosody: float, clarity: float) -> float:
    """Calculate human-likeness score."""
    try:
        # Weight different aspects
        weights = {'naturalness': 0.4, 'prosody': 0.35, 'clarity': 0.25}
        
        human_likeness = (
            naturalness * weights['naturalness'] +
            prosody * weights['prosody'] +
            clarity * weights['clarity']
        )
        
        return min(100, human_likeness)
        
    except Exception as e:
        logger.warning(f"Human-likeness calculation failed: {e}")
        return 50.0


async def analyze_simple_perceptual_quality(text: str, voice: str) -> SimplePerceptualMetrics:
    """Perform simple perceptual quality analysis."""
    try:
        logger.info(f"Analyzing: '{text}' with voice '{voice}'")
        
        # Initialize components
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        whisper_validator = get_whisper_validator("base")
        quality_validator = AudioQualityValidator()
        
        # Generate TTS audio
        tts_result = await inference_engine.generate_speech(
            text=text,
            voice=voice,
            response_format="wav"
        )
        
        # Process audio
        audio_data, audio_metadata = await audio_processor.process_audio(
            audio_array=tts_result["audio_data"],
            sample_rate=tts_result["sample_rate"],
            output_format="wav"
        )
        
        # Intelligibility analysis
        validation_result = whisper_validator.validate_tts_output(
            original_text=text,
            audio_data=audio_data,
            sample_rate=tts_result["sample_rate"]
        )
        
        accuracy_metrics = validation_result.get("accuracy_metrics", {})
        transcription_accuracy = accuracy_metrics.get("overall_accuracy", 0)
        wer = accuracy_metrics.get("wer", 1.0)
        
        # Technical quality analysis
        quality_metrics = quality_validator.analyze_audio(
            tts_result["audio_data"],
            tts_result["sample_rate"],
            tts_result.get("rtf", 0),
            tts_result.get("inference_time", 0)
        )
        
        # Simple perceptual analysis
        prosody_score = calculate_simple_prosody(
            tts_result["audio_data"], tts_result["sample_rate"], text
        )
        
        human_likeness = calculate_human_likeness(
            quality_metrics.naturalness_score,
            prosody_score,
            quality_metrics.clarity_score
        )
        
        # Create metrics
        metrics = SimplePerceptualMetrics(
            transcription_accuracy=transcription_accuracy,
            word_error_rate=wer,
            overall_quality=quality_metrics.overall_quality,
            naturalness_score=quality_metrics.naturalness_score,
            clarity_score=quality_metrics.clarity_score,
            prosody_score=prosody_score,
            human_likeness=human_likeness,
            rtf=tts_result.get("rtf", 0),
            voice=voice,
            timestamp=datetime.now().isoformat()
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        # Return minimal metrics on failure
        return SimplePerceptualMetrics(
            transcription_accuracy=0.0,
            word_error_rate=1.0,
            overall_quality=0.0,
            naturalness_score=0.0,
            clarity_score=0.0,
            prosody_score=0.0,
            human_likeness=0.0,
            rtf=0.0,
            voice=voice,
            timestamp=datetime.now().isoformat()
        )


async def main():
    """Main test execution."""
    logger.info("🧪 Starting Simple Perceptual Quality Test")
    logger.info("=" * 50)
    
    try:
        # Test case
        test_text = "Hello world, this is a test."
        test_voice = "alloy"
        
        # Perform analysis
        metrics = await analyze_simple_perceptual_quality(test_text, test_voice)
        
        # Display results
        logger.info("📊 PERCEPTUAL QUALITY RESULTS")
        logger.info("=" * 30)
        logger.info(f"Text: {test_text}")
        logger.info(f"Voice: {test_voice}")
        logger.info("")
        logger.info(f"Transcription Accuracy: {metrics.transcription_accuracy:.1f}%")
        logger.info(f"Word Error Rate: {metrics.word_error_rate:.3f}")
        logger.info(f"Overall Quality: {metrics.overall_quality:.1f}%")
        logger.info(f"Naturalness: {metrics.naturalness_score:.1f}%")
        logger.info(f"Clarity: {metrics.clarity_score:.1f}%")
        logger.info(f"Prosody Score: {metrics.prosody_score:.1f}%")
        logger.info(f"Human Likeness: {metrics.human_likeness:.1f}%")
        logger.info(f"RTF: {metrics.rtf:.3f}")
        logger.info("")
        
        # Assessment
        if metrics.transcription_accuracy < 50:
            logger.critical("❌ CRITICAL: Audio is unintelligible")
        elif metrics.transcription_accuracy < 80:
            logger.warning("⚠️  WARNING: Poor intelligibility")
        else:
            logger.info("✅ Good intelligibility")
        
        if metrics.human_likeness < 50:
            logger.warning("⚠️  WARNING: Audio sounds robotic")
        elif metrics.human_likeness >= 75:
            logger.info("✅ Good human-likeness")
        
        # Save results
        results = {
            "simple_perceptual_test": {
                "timestamp": metrics.timestamp,
                "test_case": {"text": test_text, "voice": test_voice},
                "metrics": {
                    "transcription_accuracy": float(metrics.transcription_accuracy),
                    "word_error_rate": float(metrics.word_error_rate),
                    "overall_quality": float(metrics.overall_quality),
                    "naturalness_score": float(metrics.naturalness_score),
                    "clarity_score": float(metrics.clarity_score),
                    "prosody_score": float(metrics.prosody_score),
                    "human_likeness": float(metrics.human_likeness),
                    "rtf": float(metrics.rtf)
                }
            }
        }
        
        output_file = Path("simple_perceptual_test_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {output_file}")
        logger.info("✅ Simple perceptual quality test completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
