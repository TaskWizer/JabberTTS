#!/usr/bin/env python3
"""
Comprehensive Audio Quality Diagnostic Script for JabberTTS

This script analyzes and diagnoses audio quality issues including:
1. Speed control distortion analysis
2. Phonemization impact assessment
3. Prosody and punctuation handling evaluation
4. Performance bottleneck identification
"""

import asyncio
import logging
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any, Tuple

from jabbertts.inference.engine import get_inference_engine
from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.audio.processor import get_audio_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioQualityDiagnostic:
    """Comprehensive audio quality diagnostic system."""
    
    def __init__(self):
        """Initialize diagnostic system."""
        self.inference_engine = None
        self.whisper_validator = None
        self.audio_processor = None
        self.results = {}
        
        # Test configurations
        self.test_texts = {
            "simple": "Hello world",
            "punctuation": "Hello, world! How are you today? I'm fine, thanks.",
            "numbers": "I have 123 apples and 45 oranges. That's 168 fruits total.",
            "prosody": "This is VERY important! Please listen carefully... Are you ready?",
            "long": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for testing purposes."
        }
        
        self.speed_tests = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        self.voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing audio quality diagnostic system...")
        
        self.inference_engine = get_inference_engine()
        self.whisper_validator = get_whisper_validator("base")
        self.audio_processor = get_audio_processor()
        
        logger.info("Diagnostic system initialized")
    
    async def run_comprehensive_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive audio quality diagnostic."""
        logger.info("🔍 Starting Comprehensive Audio Quality Diagnostic")
        logger.info("=" * 70)
        
        await self.initialize()
        
        # 1. Speed Control Distortion Analysis
        logger.info("\n📊 1. SPEED CONTROL DISTORTION ANALYSIS")
        speed_results = await self.analyze_speed_distortion()
        
        # 2. Phonemization Impact Assessment
        logger.info("\n📊 2. PHONEMIZATION IMPACT ASSESSMENT")
        phoneme_results = await self.analyze_phonemization_impact()
        
        # 3. Prosody and Punctuation Analysis
        logger.info("\n📊 3. PROSODY AND PUNCTUATION ANALYSIS")
        prosody_results = await self.analyze_prosody_punctuation()
        
        # 4. Performance Bottleneck Analysis
        logger.info("\n📊 4. PERFORMANCE BOTTLENECK ANALYSIS")
        performance_results = await self.analyze_performance_bottlenecks()
        
        # 5. Voice Quality Consistency Analysis
        logger.info("\n📊 5. VOICE QUALITY CONSISTENCY ANALYSIS")
        voice_results = await self.analyze_voice_consistency()
        
        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": time.time(),
            "speed_distortion": speed_results,
            "phonemization_impact": phoneme_results,
            "prosody_punctuation": prosody_results,
            "performance_bottlenecks": performance_results,
            "voice_consistency": voice_results,
            "recommendations": self.generate_recommendations()
        }
        
        # Save results
        self.save_diagnostic_results(comprehensive_results)
        
        return comprehensive_results
    
    async def analyze_speed_distortion(self) -> Dict[str, Any]:
        """Analyze speed control distortion issues."""
        logger.info("Analyzing speed control distortion...")
        
        results = {
            "speed_tests": {},
            "distortion_analysis": {},
            "quality_degradation": {}
        }
        
        test_text = self.test_texts["simple"]
        
        for speed in self.speed_tests:
            logger.info(f"Testing speed: {speed}x")
            
            try:
                # Generate audio at different speeds
                start_time = time.time()
                
                tts_result = await self.inference_engine.generate_speech(
                    text=test_text,
                    voice="alloy",
                    speed=speed,
                    response_format="wav"
                )
                
                generation_time = time.time() - start_time
                
                # Analyze audio quality
                audio_data = tts_result["audio_data"]
                sample_rate = tts_result["sample_rate"]
                
                # Calculate audio metrics
                rms = np.sqrt(np.mean(audio_data**2))
                peak = np.max(np.abs(audio_data))
                duration = len(audio_data) / sample_rate
                
                # Spectral analysis
                if len(audio_data) > 0:
                    fft = np.fft.fft(audio_data)
                    magnitude = np.abs(fft)
                    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
                    
                    # Find spectral centroid
                    spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
                else:
                    spectral_centroid = 0
                
                # Whisper validation
                validation_result = self.whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=audio_data,
                    sample_rate=sample_rate
                )
                
                accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                
                # Detect distortion indicators
                distortion_indicators = {
                    "clipping": peak >= 0.99,
                    "low_rms": rms < 0.01,
                    "high_rms": rms > 0.8,
                    "spectral_anomaly": spectral_centroid < 100 or spectral_centroid > 8000,
                    "duration_anomaly": abs(duration - len(test_text) * 0.1) > 2.0  # Rough estimate
                }
                
                results["speed_tests"][speed] = {
                    "generation_time": generation_time,
                    "audio_duration": duration,
                    "rms_level": rms,
                    "peak_level": peak,
                    "spectral_centroid": spectral_centroid,
                    "intelligibility": accuracy,
                    "distortion_indicators": distortion_indicators,
                    "rtf": generation_time / duration if duration > 0 else float('inf')
                }
                
                # Save audio sample for manual inspection
                output_file = Path("temp") / f"speed_test_{speed}x.wav"
                output_file.parent.mkdir(exist_ok=True)
                sf.write(str(output_file), audio_data, sample_rate)
                
                logger.info(f"  Speed {speed}x: RTF={generation_time/duration:.3f}, Accuracy={accuracy:.1f}%, RMS={rms:.3f}")
                
            except Exception as e:
                logger.error(f"Speed test failed for {speed}x: {e}")
                results["speed_tests"][speed] = {"error": str(e)}
        
        # Analyze distortion patterns
        results["distortion_analysis"] = self.analyze_distortion_patterns(results["speed_tests"])
        
        return results
    
    async def analyze_phonemization_impact(self) -> Dict[str, Any]:
        """Analyze the impact of phonemization on audio quality."""
        logger.info("Analyzing phonemization impact...")
        
        results = {
            "with_phonemization": {},
            "without_phonemization": {},
            "comparison": {}
        }
        
        for text_type, text in self.test_texts.items():
            logger.info(f"Testing phonemization with: {text_type}")
            
            # Test with phonemization enabled
            try:
                self.inference_engine.preprocessor.use_phonemizer = True
                
                result_with = await self.inference_engine.generate_speech(
                    text=text,
                    voice="alloy",
                    speed=1.0,
                    response_format="wav"
                )
                
                validation_with = self.whisper_validator.validate_tts_output(
                    original_text=text,
                    audio_data=result_with["audio_data"],
                    sample_rate=result_with["sample_rate"]
                )
                
                results["with_phonemization"][text_type] = {
                    "accuracy": validation_with.get("accuracy_metrics", {}).get("overall_accuracy", 0),
                    "transcription": validation_with.get("transcription", ""),
                    "audio_duration": len(result_with["audio_data"]) / result_with["sample_rate"]
                }
                
            except Exception as e:
                logger.error(f"Phonemization test failed: {e}")
                results["with_phonemization"][text_type] = {"error": str(e)}
            
            # Test without phonemization
            try:
                self.inference_engine.preprocessor.use_phonemizer = False
                
                result_without = await self.inference_engine.generate_speech(
                    text=text,
                    voice="alloy",
                    speed=1.0,
                    response_format="wav"
                )
                
                validation_without = self.whisper_validator.validate_tts_output(
                    original_text=text,
                    audio_data=result_without["audio_data"],
                    sample_rate=result_without["sample_rate"]
                )
                
                results["without_phonemization"][text_type] = {
                    "accuracy": validation_without.get("accuracy_metrics", {}).get("overall_accuracy", 0),
                    "transcription": validation_without.get("transcription", ""),
                    "audio_duration": len(result_without["audio_data"]) / result_without["sample_rate"]
                }
                
            except Exception as e:
                logger.error(f"Non-phonemization test failed: {e}")
                results["without_phonemization"][text_type] = {"error": str(e)}
            
            # Compare results
            if text_type in results["with_phonemization"] and text_type in results["without_phonemization"]:
                with_acc = results["with_phonemization"][text_type].get("accuracy", 0)
                without_acc = results["without_phonemization"][text_type].get("accuracy", 0)
                
                results["comparison"][text_type] = {
                    "accuracy_improvement": without_acc - with_acc,
                    "better_without_phonemization": without_acc > with_acc,
                    "significant_difference": abs(without_acc - with_acc) > 10
                }
                
                logger.info(f"  {text_type}: With={with_acc:.1f}%, Without={without_acc:.1f}%, Improvement={without_acc-with_acc:.1f}%")
        
        return results
    
    async def analyze_prosody_punctuation(self) -> Dict[str, Any]:
        """Analyze prosody and punctuation handling."""
        logger.info("Analyzing prosody and punctuation handling...")
        
        results = {
            "punctuation_tests": {},
            "prosody_tests": {},
            "pause_analysis": {}
        }
        
        # Test punctuation handling
        punctuation_texts = {
            "no_punctuation": "Hello world how are you today",
            "basic_punctuation": "Hello, world! How are you today?",
            "complex_punctuation": "Hello... world? How are you today; I'm fine, thanks!",
            "questions": "Are you ready? Can you hear me? What's your name?",
            "exclamations": "This is amazing! Wow! Incredible!"
        }
        
        for test_name, text in punctuation_texts.items():
            try:
                result = await self.inference_engine.generate_speech(
                    text=text,
                    voice="alloy",
                    speed=1.0,
                    response_format="wav"
                )
                
                # Analyze pause patterns
                audio_data = result["audio_data"]
                pause_analysis = self.analyze_pause_patterns(audio_data, result["sample_rate"])
                
                validation = self.whisper_validator.validate_tts_output(
                    original_text=text,
                    audio_data=audio_data,
                    sample_rate=result["sample_rate"]
                )
                
                results["punctuation_tests"][test_name] = {
                    "accuracy": validation.get("accuracy_metrics", {}).get("overall_accuracy", 0),
                    "pause_analysis": pause_analysis,
                    "transcription": validation.get("transcription", "")
                }
                
                logger.info(f"  {test_name}: Accuracy={validation.get('accuracy_metrics', {}).get('overall_accuracy', 0):.1f}%")
                
            except Exception as e:
                logger.error(f"Punctuation test failed for {test_name}: {e}")
                results["punctuation_tests"][test_name] = {"error": str(e)}
        
        return results
    
    async def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        logger.info("Analyzing performance bottlenecks...")
        
        results = {
            "timing_breakdown": {},
            "memory_usage": {},
            "rtf_analysis": {}
        }
        
        test_text = self.test_texts["long"]
        
        # Detailed timing analysis
        start_total = time.time()
        
        # Preprocessing timing
        start_preprocess = time.time()
        processed_text = self.inference_engine.preprocessor.preprocess(test_text)
        preprocess_time = time.time() - start_preprocess
        
        # Model loading timing
        start_model = time.time()
        model = await self.inference_engine._ensure_model_loaded()
        model_load_time = time.time() - start_model
        
        # Inference timing
        start_inference = time.time()
        audio_data = await self.inference_engine._generate_audio(model, processed_text, "alloy", 1.0)
        inference_time = time.time() - start_inference
        
        total_time = time.time() - start_total
        audio_duration = len(audio_data) / model.get_sample_rate()
        
        results["timing_breakdown"] = {
            "preprocessing_time": preprocess_time,
            "model_loading_time": model_load_time,
            "inference_time": inference_time,
            "total_time": total_time,
            "audio_duration": audio_duration,
            "rtf": total_time / audio_duration if audio_duration > 0 else float('inf')
        }
        
        # Memory usage analysis
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            results["memory_usage"] = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent()
            }
        except ImportError:
            results["memory_usage"] = {"error": "psutil not available"}
        
        logger.info(f"  RTF: {results['timing_breakdown']['rtf']:.3f}")
        logger.info(f"  Memory: {results['memory_usage'].get('rss_mb', 0):.1f}MB")
        
        return results
    
    async def analyze_voice_consistency(self) -> Dict[str, Any]:
        """Analyze consistency across different voices."""
        logger.info("Analyzing voice consistency...")
        
        results = {}
        test_text = self.test_texts["simple"]
        
        for voice in self.voices:
            try:
                result = await self.inference_engine.generate_speech(
                    text=test_text,
                    voice=voice,
                    speed=1.0,
                    response_format="wav"
                )
                
                validation = self.whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=result["audio_data"],
                    sample_rate=result["sample_rate"]
                )
                
                # Audio quality metrics
                audio_data = result["audio_data"]
                rms = np.sqrt(np.mean(audio_data**2))
                peak = np.max(np.abs(audio_data))
                
                results[voice] = {
                    "accuracy": validation.get("accuracy_metrics", {}).get("overall_accuracy", 0),
                    "rms_level": rms,
                    "peak_level": peak,
                    "duration": len(audio_data) / result["sample_rate"],
                    "transcription": validation.get("transcription", "")
                }
                
                logger.info(f"  {voice}: Accuracy={results[voice]['accuracy']:.1f}%, RMS={rms:.3f}")
                
            except Exception as e:
                logger.error(f"Voice test failed for {voice}: {e}")
                results[voice] = {"error": str(e)}
        
        return results
    
    def analyze_distortion_patterns(self, speed_tests: Dict) -> Dict[str, Any]:
        """Analyze distortion patterns across speed tests."""
        patterns = {
            "distortion_speeds": [],
            "quality_degradation_threshold": None,
            "optimal_speed_range": []
        }
        
        for speed, data in speed_tests.items():
            if "error" in data:
                continue
            
            # Check for distortion indicators
            distortion_count = sum(data.get("distortion_indicators", {}).values())
            if distortion_count > 0:
                patterns["distortion_speeds"].append(speed)
            
            # Check for quality degradation
            if data.get("intelligibility", 0) < 50:
                if patterns["quality_degradation_threshold"] is None:
                    patterns["quality_degradation_threshold"] = speed
            
            # Identify optimal range
            if (data.get("intelligibility", 0) > 80 and 
                data.get("rtf", float('inf')) < 1.0 and
                distortion_count == 0):
                patterns["optimal_speed_range"].append(speed)
        
        return patterns
    
    def analyze_pause_patterns(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze pause patterns in audio."""
        # Simple silence detection
        silence_threshold = 0.01
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        
        frames = []
        for i in range(0, len(audio_data) - frame_length, frame_length):
            frame = audio_data[i:i + frame_length]
            rms = np.sqrt(np.mean(frame**2))
            frames.append(rms > silence_threshold)
        
        # Find pause segments
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, is_speech in enumerate(frames):
            if not is_speech and not in_pause:
                in_pause = True
                pause_start = i
            elif is_speech and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * frame_length / sample_rate
                if pause_duration > 0.1:  # Only count pauses > 100ms
                    pauses.append(pause_duration)
        
        return {
            "total_pauses": len(pauses),
            "average_pause_duration": np.mean(pauses) if pauses else 0,
            "max_pause_duration": np.max(pauses) if pauses else 0,
            "pause_durations": pauses
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = [
            "1. CRITICAL: Fix speed control distortion by implementing proper time-stretching algorithms",
            "2. CRITICAL: Disable phonemization for SpeechT5 model to improve intelligibility",
            "3. Enhance punctuation handling for better prosody and natural pauses",
            "4. Optimize preprocessing pipeline to reduce RTF and improve performance",
            "5. Implement voice-specific quality adjustments for consistency",
            "6. Add comprehensive error handling and fallback mechanisms",
            "7. Increase resource utilization for better performance targets"
        ]
        return recommendations
    
    def save_diagnostic_results(self, results: Dict[str, Any]) -> None:
        """Save diagnostic results to file."""
        import json
        
        output_file = Path("temp") / "audio_quality_diagnostic_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Diagnostic results saved to: {output_file}")


async def main():
    """Main execution function."""
    diagnostic = AudioQualityDiagnostic()
    results = await diagnostic.run_comprehensive_diagnostic()
    
    logger.info("\n🎯 DIAGNOSTIC COMPLETE")
    logger.info("=" * 50)
    
    # Print key findings
    speed_results = results.get("speed_distortion", {})
    if "distortion_analysis" in speed_results:
        distortion_speeds = speed_results["distortion_analysis"].get("distortion_speeds", [])
        if distortion_speeds:
            logger.info(f"⚠️ Speed distortion detected at: {distortion_speeds}")
    
    phoneme_results = results.get("phonemization_impact", {})
    if "comparison" in phoneme_results:
        improvements = [
            data.get("accuracy_improvement", 0) 
            for data in phoneme_results["comparison"].values()
        ]
        avg_improvement = np.mean(improvements) if improvements else 0
        logger.info(f"📈 Average intelligibility improvement without phonemization: {avg_improvement:.1f}%")
    
    performance_results = results.get("performance_bottlenecks", {})
    if "timing_breakdown" in performance_results:
        rtf = performance_results["timing_breakdown"].get("rtf", 0)
        logger.info(f"⚡ Current RTF: {rtf:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
