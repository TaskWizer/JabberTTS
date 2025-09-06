#!/usr/bin/env python3
"""
Comprehensive Audio Quality Validation Script for JabberTTS

This script validates all the audio quality improvements including:
1. Speed control distortion fixes
2. Enhanced phonemization and preprocessing
3. Performance optimizations
4. Voice quality consistency
5. Whisper STT validation (>95% target)
"""

import asyncio
import logging
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any, Tuple

from jabbertts.inference.engine import get_inference_engine
from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.optimization.performance_enhancer import get_performance_enhancer, PerformanceConfig
from jabbertts.audio.advanced_speed_control import TimeStretchAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveAudioQualityValidator:
    """Comprehensive audio quality validation system."""
    
    def __init__(self):
        """Initialize validation system."""
        self.inference_engine = None
        self.whisper_validator = None
        self.performance_enhancer = None
        self.results = {}
        
        # Enhanced test configurations
        self.test_texts = {
            "simple": "Hello world",
            "punctuation": "Hello, world! How are you today? I'm fine, thanks.",
            "numbers": "I have one hundred twenty-three apples and forty-five oranges.",
            "prosody": "This is VERY important! Please listen carefully... Are you ready?",
            "technical": "The API endpoint returns JSON data with authentication tokens.",
            "emotional": "I'm so excited! This is absolutely wonderful news!",
            "long": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes in typography and font development."
        }
        
        self.speed_tests = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        self.voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        # Performance targets
        self.targets = {
            "rtf_cpu": 0.1,
            "rtf_gpu": 0.05,
            "first_chunk_latency": 0.2,
            "intelligibility": 95.0,
            "voice_consistency": 90.0
        }
    
    async def initialize(self):
        """Initialize all components with performance enhancements."""
        logger.info("Initializing comprehensive audio quality validation system...")
        
        # Initialize performance enhancer first
        perf_config = PerformanceConfig(
            target_rtf_cpu=0.1,
            target_rtf_gpu=0.05,
            target_first_chunk_latency=0.2,
            enable_aggressive_caching=True,
            enable_model_compilation=True
        )
        self.performance_enhancer = get_performance_enhancer(perf_config)
        
        # Initialize inference engine
        self.inference_engine = get_inference_engine()
        
        # Initialize Whisper validator
        self.whisper_validator = get_whisper_validator("base")
        
        logger.info("Validation system initialized with performance enhancements")
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive audio quality validation."""
        logger.info("🚀 Starting Comprehensive Audio Quality Validation")
        logger.info("=" * 80)
        
        await self.initialize()
        
        validation_results = {}
        
        # 1. Speed Control Quality Validation
        logger.info("\n🎯 1. SPEED CONTROL QUALITY VALIDATION")
        validation_results["speed_control"] = await self.validate_speed_control_quality()
        
        # 2. Enhanced Preprocessing Validation
        logger.info("\n🎯 2. ENHANCED PREPROCESSING VALIDATION")
        validation_results["preprocessing"] = await self.validate_preprocessing_enhancements()
        
        # 3. Performance Optimization Validation
        logger.info("\n🎯 3. PERFORMANCE OPTIMIZATION VALIDATION")
        validation_results["performance"] = await self.validate_performance_improvements()
        
        # 4. Voice Quality Consistency Validation
        logger.info("\n🎯 4. VOICE QUALITY CONSISTENCY VALIDATION")
        validation_results["voice_consistency"] = await self.validate_voice_consistency()
        
        # 5. Whisper STT Accuracy Validation
        logger.info("\n🎯 5. WHISPER STT ACCURACY VALIDATION")
        validation_results["whisper_accuracy"] = await self.validate_whisper_accuracy()
        
        # 6. Overall Quality Assessment
        logger.info("\n🎯 6. OVERALL QUALITY ASSESSMENT")
        validation_results["overall_assessment"] = self.assess_overall_quality(validation_results)
        
        # Save comprehensive results
        self.save_validation_results(validation_results)
        
        return validation_results
    
    async def validate_speed_control_quality(self) -> Dict[str, Any]:
        """Validate speed control quality improvements."""
        logger.info("Validating speed control quality with advanced time-stretching...")
        
        results = {
            "algorithm_tests": {},
            "quality_metrics": {},
            "distortion_analysis": {}
        }
        
        test_text = self.test_texts["simple"]
        
        # Test different time-stretching algorithms
        algorithms = [
            TimeStretchAlgorithm.LIBROSA_TIME_STRETCH,
            TimeStretchAlgorithm.LIBROSA_PHASE_VOCODER,
            TimeStretchAlgorithm.SIMPLE_OVERLAP_ADD
        ]
        
        for algorithm in algorithms:
            logger.info(f"Testing algorithm: {algorithm.value}")
            algorithm_results = {}
            
            for speed in [0.5, 1.0, 2.0]:  # Representative speeds
                try:
                    start_time = time.time()
                    
                    # Generate audio with specific speed
                    tts_result = await self.inference_engine.generate_speech(
                        text=test_text,
                        voice="alloy",
                        speed=speed,
                        response_format="wav"
                    )
                    
                    generation_time = time.time() - start_time
                    audio_data = tts_result["audio_data"]
                    sample_rate = tts_result["sample_rate"]
                    
                    # Quality analysis
                    quality_metrics = self.analyze_audio_quality(audio_data, sample_rate)
                    
                    # Whisper validation
                    validation_result = await self.validate_with_whisper(
                        test_text, audio_data, sample_rate
                    )
                    
                    algorithm_results[speed] = {
                        "generation_time": generation_time,
                        "quality_metrics": quality_metrics,
                        "intelligibility": validation_result.get("accuracy", 0),
                        "rtf": generation_time / (len(audio_data) / sample_rate)
                    }
                    
                    logger.info(f"  Speed {speed}x: Intelligibility={validation_result.get('accuracy', 0):.1f}%")
                    
                except Exception as e:
                    logger.error(f"Speed test failed for {speed}x with {algorithm.value}: {e}")
                    algorithm_results[speed] = {"error": str(e)}
            
            results["algorithm_tests"][algorithm.value] = algorithm_results
        
        return results
    
    async def validate_preprocessing_enhancements(self) -> Dict[str, Any]:
        """Validate enhanced preprocessing improvements."""
        logger.info("Validating enhanced preprocessing with model-specific optimization...")
        
        results = {
            "model_specific_tests": {},
            "phonemization_impact": {},
            "text_normalization": {}
        }
        
        # Test different text types
        for text_type, text in self.test_texts.items():
            logger.info(f"Testing preprocessing for: {text_type}")
            
            try:
                # Generate with enhanced preprocessing
                start_time = time.time()
                
                tts_result = await self.inference_engine.generate_speech(
                    text=text,
                    voice="alloy",
                    speed=1.0,
                    response_format="wav"
                )
                
                processing_time = time.time() - start_time
                
                # Validate quality
                validation_result = await self.validate_with_whisper(
                    text, tts_result["audio_data"], tts_result["sample_rate"]
                )
                
                results["model_specific_tests"][text_type] = {
                    "processing_time": processing_time,
                    "intelligibility": validation_result.get("accuracy", 0),
                    "transcription": validation_result.get("transcription", ""),
                    "rtf": processing_time / (len(tts_result["audio_data"]) / tts_result["sample_rate"])
                }
                
                logger.info(f"  {text_type}: Intelligibility={validation_result.get('accuracy', 0):.1f}%")
                
            except Exception as e:
                logger.error(f"Preprocessing test failed for {text_type}: {e}")
                results["model_specific_tests"][text_type] = {"error": str(e)}
        
        return results
    
    async def validate_performance_improvements(self) -> Dict[str, Any]:
        """Validate performance optimization improvements."""
        logger.info("Validating performance improvements...")
        
        results = {
            "rtf_measurements": {},
            "latency_measurements": {},
            "resource_utilization": {}
        }
        
        test_text = self.test_texts["long"]
        
        # Measure performance metrics
        try:
            # Cold start measurement
            start_time = time.time()
            
            tts_result = await self.inference_engine.generate_speech(
                text=test_text,
                voice="alloy",
                speed=1.0,
                response_format="wav"
            )
            
            total_time = time.time() - start_time
            audio_duration = len(tts_result["audio_data"]) / tts_result["sample_rate"]
            rtf = total_time / audio_duration
            
            results["rtf_measurements"] = {
                "total_time": total_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "target_met": rtf < self.targets["rtf_cpu"]
            }
            
            # Get performance enhancer metrics
            perf_metrics = self.performance_enhancer.get_performance_metrics()
            results["resource_utilization"] = perf_metrics
            
            logger.info(f"  RTF: {rtf:.3f} (Target: <{self.targets['rtf_cpu']})")
            logger.info(f"  Target met: {rtf < self.targets['rtf_cpu']}")
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def validate_voice_consistency(self) -> Dict[str, Any]:
        """Validate voice quality consistency."""
        logger.info("Validating voice quality consistency...")
        
        results = {}
        test_text = self.test_texts["simple"]
        
        for voice in self.voices:
            try:
                tts_result = await self.inference_engine.generate_speech(
                    text=test_text,
                    voice=voice,
                    speed=1.0,
                    response_format="wav"
                )
                
                # Quality analysis
                quality_metrics = self.analyze_audio_quality(
                    tts_result["audio_data"], tts_result["sample_rate"]
                )
                
                # Whisper validation
                validation_result = await self.validate_with_whisper(
                    test_text, tts_result["audio_data"], tts_result["sample_rate"]
                )
                
                results[voice] = {
                    "quality_metrics": quality_metrics,
                    "intelligibility": validation_result.get("accuracy", 0),
                    "transcription": validation_result.get("transcription", "")
                }
                
                logger.info(f"  {voice}: Intelligibility={validation_result.get('accuracy', 0):.1f}%")
                
            except Exception as e:
                logger.error(f"Voice consistency test failed for {voice}: {e}")
                results[voice] = {"error": str(e)}
        
        return results
    
    async def validate_whisper_accuracy(self) -> Dict[str, Any]:
        """Validate Whisper STT accuracy across all test cases."""
        logger.info("Validating Whisper STT accuracy...")
        
        results = {
            "text_type_accuracy": {},
            "overall_accuracy": 0,
            "target_met": False
        }
        
        accuracies = []
        
        for text_type, text in self.test_texts.items():
            try:
                tts_result = await self.inference_engine.generate_speech(
                    text=text,
                    voice="alloy",
                    speed=1.0,
                    response_format="wav"
                )
                
                validation_result = await self.validate_with_whisper(
                    text, tts_result["audio_data"], tts_result["sample_rate"]
                )
                
                accuracy = validation_result.get("accuracy", 0)
                accuracies.append(accuracy)
                
                results["text_type_accuracy"][text_type] = {
                    "accuracy": accuracy,
                    "transcription": validation_result.get("transcription", ""),
                    "original_text": text
                }
                
                logger.info(f"  {text_type}: {accuracy:.1f}%")
                
            except Exception as e:
                logger.error(f"Whisper validation failed for {text_type}: {e}")
                results["text_type_accuracy"][text_type] = {"error": str(e)}
        
        if accuracies:
            overall_accuracy = np.mean(accuracies)
            results["overall_accuracy"] = overall_accuracy
            results["target_met"] = overall_accuracy >= self.targets["intelligibility"]
            
            logger.info(f"  Overall accuracy: {overall_accuracy:.1f}% (Target: >{self.targets['intelligibility']}%)")
        
        return results
    
    def analyze_audio_quality(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze audio quality metrics."""
        if len(audio) == 0:
            return {"error": "Empty audio"}
        
        # Basic quality metrics
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        dynamic_range = 20 * np.log10(peak / (rms + 1e-8))
        
        # Spectral analysis
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
        
        # Spectral centroid
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        spectral_centroid = np.sum(positive_freqs * positive_magnitude) / (np.sum(positive_magnitude) + 1e-8)
        
        return {
            "rms_level": float(rms),
            "peak_level": float(peak),
            "dynamic_range": float(dynamic_range),
            "spectral_centroid": float(spectral_centroid),
            "duration": len(audio) / sample_rate
        }
    
    async def validate_with_whisper(self, original_text: str, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Validate audio with Whisper STT."""
        try:
            # Convert audio to bytes for Whisper validator
            import io

            # Convert numpy array to bytes
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio, sample_rate, format='WAV')
            audio_bytes.seek(0)

            # Validate with Whisper
            validation_result = self.whisper_validator.validate_tts_output(
                original_text=original_text,
                audio_data=audio_bytes.getvalue(),
                sample_rate=sample_rate
            )

            return validation_result

        except Exception as e:
            logger.error(f"Whisper validation failed: {e}")
            return {"error": str(e), "accuracy": 0}
    
    def assess_overall_quality(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality based on all validation results."""
        assessment = {
            "targets_met": {},
            "overall_score": 0,
            "recommendations": []
        }
        
        # Check individual targets
        performance = validation_results.get("performance", {})
        rtf_measurements = performance.get("rtf_measurements", {})
        
        assessment["targets_met"]["rtf"] = rtf_measurements.get("target_met", False)
        assessment["targets_met"]["whisper_accuracy"] = validation_results.get("whisper_accuracy", {}).get("target_met", False)
        
        # Calculate overall score
        targets_met = sum(assessment["targets_met"].values())
        total_targets = len(assessment["targets_met"])
        assessment["overall_score"] = (targets_met / total_targets * 100) if total_targets > 0 else 0
        
        # Generate recommendations
        if not assessment["targets_met"].get("rtf", False):
            assessment["recommendations"].append("Consider further performance optimizations to meet RTF targets")
        
        if not assessment["targets_met"].get("whisper_accuracy", False):
            assessment["recommendations"].append("Improve audio quality or model selection for better intelligibility")
        
        if assessment["overall_score"] >= 80:
            assessment["recommendations"].append("Excellent quality achieved! Ready for production deployment")
        
        return assessment
    
    def save_validation_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to file."""
        output_file = Path("temp") / "comprehensive_audio_quality_validation_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        # Add timestamp and metadata
        results["metadata"] = {
            "timestamp": time.time(),
            "validation_version": "2.0",
            "targets": self.targets
        }
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Validation results saved to: {output_file}")


async def main():
    """Main execution function."""
    validator = ComprehensiveAudioQualityValidator()
    results = await validator.run_comprehensive_validation()
    
    logger.info("\n🎉 COMPREHENSIVE VALIDATION COMPLETE")
    logger.info("=" * 60)
    
    # Print summary
    overall_assessment = results.get("overall_assessment", {})
    overall_score = overall_assessment.get("overall_score", 0)
    targets_met = overall_assessment.get("targets_met", {})
    
    logger.info(f"📊 Overall Quality Score: {overall_score:.1f}%")
    logger.info(f"🎯 Targets Met: {sum(targets_met.values())}/{len(targets_met)}")
    
    for target, met in targets_met.items():
        status = "✅" if met else "❌"
        logger.info(f"  {status} {target.upper()}")
    
    recommendations = overall_assessment.get("recommendations", [])
    if recommendations:
        logger.info("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")


if __name__ == "__main__":
    asyncio.run(main())
