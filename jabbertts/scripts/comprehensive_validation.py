#!/usr/bin/env python3
"""
Comprehensive TTS Validation Script

This script performs systematic validation of the current TTS system across:
1. All 6 voices (alloy, echo, fable, onyx, nova, shimmer)
2. Multiple text lengths (10, 25, 50, 100, 200, 500, 1000+ characters)
3. Whisper STT validation with detailed metrics
4. Identification of exact degradation thresholds

Usage:
    python jabbertts/scripts/comprehensive_validation.py
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a single validation test."""
    voice: str
    text: str
    text_length: int
    transcription: str
    accuracy: float
    wer: float
    cer: float
    rtf: float
    processing_time: float
    audio_duration: float
    status: str
    has_artifacts: bool
    artifact_description: str


class ComprehensiveValidator:
    """Comprehensive TTS validation system."""
    
    def __init__(self):
        """Initialize the validator."""
        self.inference_engine = None
        self.audio_processor = None
        self.whisper_validator = None
        self.results = []
        
        # Test texts of different lengths
        self.test_texts = {
            10: "Hello!",
            25: "This is a short test.",
            50: "The quick brown fox jumps over the lazy dog.",
            100: "This is a longer test sentence that contains exactly one hundred characters for testing purposes.",
            200: "This is a much longer test sentence that is designed to test the text-to-speech system with a moderate amount of text. It should contain approximately two hundred characters to test medium-length inputs.",
            500: "This is an even longer test passage that is specifically designed to evaluate the performance of the text-to-speech system when processing longer inputs. The purpose of this test is to identify at what point the system begins to degrade in quality or intelligibility. This text contains approximately five hundred characters and should help us understand the limitations of the current implementation when dealing with more substantial amounts of text input.",
            1000: "This is a very long test passage that is specifically designed to thoroughly evaluate the performance and limitations of the text-to-speech system when processing extended inputs. The primary purpose of this comprehensive test is to systematically identify the exact point at which the system begins to exhibit degradation in audio quality, intelligibility, or the presence of artifacts such as repetitive sounds or unintelligible output. This particular test text has been carefully crafted to contain approximately one thousand characters, which should provide sufficient content to stress-test the system and reveal any underlying issues with longer text processing. By using this extended passage, we can determine whether the system maintains consistent performance across varying input lengths or if there are specific thresholds where quality begins to deteriorate significantly."
        }
        
        # All OpenAI-compatible voices
        self.voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing comprehensive validation system...")
        
        self.inference_engine = get_inference_engine()
        self.audio_processor = get_audio_processor()
        self.whisper_validator = get_whisper_validator("base")
        
        logger.info("All components initialized successfully")
    
    def detect_artifacts(self, transcription: str, original_text: str) -> Tuple[bool, str]:
        """Detect common TTS artifacts in transcription."""
        artifacts = []
        
        # Check for repetitive patterns
        if "nan" in transcription.lower():
            artifacts.append("nan-repetition")
        
        # Check for excessive repetition
        words = transcription.lower().split()
        if len(words) > 3:
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    artifacts.append(f"word-repetition: {words[i]}")
                    break
        
        # Check for empty or very short transcription
        if len(transcription.strip()) < len(original_text) * 0.1:
            artifacts.append("severely-truncated")
        
        # Check for completely empty transcription
        if not transcription.strip():
            artifacts.append("silent-output")
        
        has_artifacts = len(artifacts) > 0
        artifact_description = "; ".join(artifacts) if artifacts else "none"
        
        return has_artifacts, artifact_description
    
    async def validate_single_case(self, voice: str, text_length: int, text: str) -> ValidationResult:
        """Validate a single test case."""
        logger.info(f"Testing voice '{voice}' with {text_length} characters: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        start_time = time.time()
        
        try:
            # Generate TTS audio
            result = await self.inference_engine.generate_speech(
                text=text,
                voice=voice,
                response_format="wav"
            )
            
            # Process audio
            audio_data, audio_metadata = await self.audio_processor.process_audio(
                audio_array=result["audio_data"],
                sample_rate=result["sample_rate"],
                output_format="wav"
            )
            
            processing_time = time.time() - start_time
            
            # Transcribe with Whisper
            validation_result = self.whisper_validator.validate_tts_output(
                original_text=text,
                audio_data=audio_data,
                sample_rate=result["sample_rate"]
            )
            
            # Extract metrics
            accuracy_metrics = validation_result.get("accuracy_metrics", {})
            accuracy = accuracy_metrics.get("overall_accuracy", 0)
            wer = accuracy_metrics.get("wer", 1.0)
            cer = accuracy_metrics.get("cer", 1.0)
            transcription = validation_result.get("transcription", "")
            
            # Detect artifacts
            has_artifacts, artifact_description = self.detect_artifacts(transcription, text)
            
            # Determine status
            if accuracy >= 95:
                status = "🟢 EXCELLENT"
            elif accuracy >= 80:
                status = "🟡 GOOD"
            elif accuracy >= 50:
                status = "🟠 POOR"
            else:
                status = "🔴 UNINTELLIGIBLE"
            
            # Calculate audio duration
            audio_duration = len(result["audio_data"]) / result["sample_rate"]
            
            return ValidationResult(
                voice=voice,
                text=text,
                text_length=text_length,
                transcription=transcription,
                accuracy=accuracy,
                wer=wer,
                cer=cer,
                rtf=result.get("rtf", 0),
                processing_time=processing_time,
                audio_duration=audio_duration,
                status=status,
                has_artifacts=has_artifacts,
                artifact_description=artifact_description
            )
            
        except Exception as e:
            logger.error(f"Validation failed for voice '{voice}', length {text_length}: {e}")
            return ValidationResult(
                voice=voice,
                text=text,
                text_length=text_length,
                transcription="",
                accuracy=0,
                wer=1.0,
                cer=1.0,
                rtf=0,
                processing_time=time.time() - start_time,
                audio_duration=0,
                status="🔴 ERROR",
                has_artifacts=True,
                artifact_description=f"error: {str(e)}"
            )
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across all voices and text lengths."""
        logger.info("🔍 Starting Comprehensive TTS Validation")
        logger.info("=" * 60)
        
        await self.initialize()
        
        all_results = []
        
        # Test each voice with each text length
        for voice in self.voices:
            logger.info(f"\n🎤 Testing voice: {voice}")
            logger.info("-" * 40)
            
            voice_results = []
            
            for text_length in sorted(self.test_texts.keys()):
                text = self.test_texts[text_length]
                result = await self.validate_single_case(voice, text_length, text)
                voice_results.append(result)
                all_results.append(result)
                
                # Log immediate results
                logger.info(f"  {text_length:4d} chars: {result.status} (Accuracy: {result.accuracy:.1f}%, RTF: {result.rtf:.3f})")
                if result.has_artifacts:
                    logger.info(f"    Artifacts: {result.artifact_description}")
                
                # Early termination if severe degradation detected
                if result.accuracy < 5 and text_length >= 100:
                    logger.warning(f"  Severe degradation detected at {text_length} characters, skipping longer texts for this voice")
                    break
        
        # Analyze results
        analysis = self.analyze_results(all_results)
        
        # Save results
        timestamp = datetime.now().isoformat()
        comprehensive_results = {
            "comprehensive_validation": {
                "timestamp": timestamp,
                "summary": analysis,
                "individual_results": [
                    {
                        "voice": r.voice,
                        "text_length": r.text_length,
                        "text": r.text,
                        "transcription": r.transcription,
                        "accuracy": r.accuracy,
                        "wer": r.wer,
                        "cer": r.cer,
                        "rtf": r.rtf,
                        "processing_time": r.processing_time,
                        "audio_duration": r.audio_duration,
                        "status": r.status,
                        "has_artifacts": r.has_artifacts,
                        "artifact_description": r.artifact_description
                    }
                    for r in all_results
                ]
            }
        }
        
        output_file = Path("temp") / "comprehensive_validation_results.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"\n📁 Results saved to: {output_file}")
        
        return comprehensive_results
    
    def analyze_results(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze validation results to identify patterns and thresholds."""
        logger.info("\n📊 Analyzing Results")
        logger.info("=" * 40)
        
        # Group by voice
        by_voice = {}
        for result in results:
            if result.voice not in by_voice:
                by_voice[result.voice] = []
            by_voice[result.voice].append(result)
        
        # Analyze degradation thresholds
        degradation_thresholds = {}
        for voice, voice_results in by_voice.items():
            threshold = None
            for result in sorted(voice_results, key=lambda x: x.text_length):
                if result.accuracy < 50:  # Below 50% accuracy
                    threshold = result.text_length
                    break
            degradation_thresholds[voice] = threshold
        
        # Overall statistics
        successful_results = [r for r in results if r.accuracy > 0]
        if successful_results:
            avg_accuracy = sum(r.accuracy for r in successful_results) / len(successful_results)
            max_accuracy = max(r.accuracy for r in successful_results)
            avg_rtf = sum(r.rtf for r in successful_results) / len(successful_results)
        else:
            avg_accuracy = max_accuracy = avg_rtf = 0
        
        # Artifact analysis
        artifact_count = sum(1 for r in results if r.has_artifacts)
        
        analysis = {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "average_accuracy": avg_accuracy,
            "maximum_accuracy": max_accuracy,
            "average_rtf": avg_rtf,
            "degradation_thresholds": degradation_thresholds,
            "artifact_rate": artifact_count / len(results) if results else 0,
            "voices_tested": len(by_voice),
            "text_lengths_tested": len(set(r.text_length for r in results))
        }
        
        # Log analysis
        logger.info(f"Total tests: {analysis['total_tests']}")
        logger.info(f"Successful tests: {analysis['successful_tests']}")
        logger.info(f"Average accuracy: {analysis['average_accuracy']:.1f}%")
        logger.info(f"Maximum accuracy: {analysis['maximum_accuracy']:.1f}%")
        logger.info(f"Average RTF: {analysis['average_rtf']:.3f}")
        logger.info(f"Artifact rate: {analysis['artifact_rate']:.1%}")
        
        logger.info("\nDegradation thresholds by voice:")
        for voice, threshold in degradation_thresholds.items():
            if threshold:
                logger.info(f"  {voice}: {threshold} characters")
            else:
                logger.info(f"  {voice}: No degradation detected")
        
        return analysis


async def main():
    """Main execution function."""
    validator = ComprehensiveValidator()
    results = await validator.run_comprehensive_validation()
    
    summary = results["comprehensive_validation"]["summary"]
    
    logger.info(f"\n🎯 VALIDATION COMPLETE")
    logger.info(f"Maximum accuracy achieved: {summary['maximum_accuracy']:.1f}%")
    logger.info(f"Average RTF: {summary['average_rtf']:.3f}")
    
    if summary['maximum_accuracy'] >= 80:
        logger.info("✅ System shows good intelligibility")
    elif summary['maximum_accuracy'] >= 50:
        logger.info("⚠️ System shows partial intelligibility")
    else:
        logger.info("❌ System shows poor intelligibility - investigation needed")


if __name__ == "__main__":
    asyncio.run(main())
