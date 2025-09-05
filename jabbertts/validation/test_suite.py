"""Automated Testing Pipeline for TTS Validation.

This module provides comprehensive test suites with diverse text samples
for automated regression testing of all voice models.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.validation.quality_assessor import QualityAssessor
from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor

logger = logging.getLogger(__name__)


class ValidationTestSuite:
    """Comprehensive automated testing pipeline for TTS validation."""
    
    def __init__(self, whisper_model_size: str = "base"):
        """Initialize validation test suite.
        
        Args:
            whisper_model_size: Whisper model size for validation
        """
        self.whisper_model_size = whisper_model_size
        self.validator = get_whisper_validator(whisper_model_size)
        self.quality_assessor = QualityAssessor()
        self.test_samples = self._load_test_samples()
        
        logger.info(f"Validation test suite initialized with Whisper {whisper_model_size}")
    
    def _load_test_samples(self) -> List[Dict[str, Any]]:
        """Load diverse test samples for validation.
        
        Returns:
            List of test sample dictionaries
        """
        return [
            # Basic pronunciation tests
            {
                "category": "pronunciation",
                "text": "The quick brown fox jumps over the lazy dog.",
                "expected_difficulty": "easy",
                "target_accuracy": 0.95
            },
            {
                "category": "pronunciation", 
                "text": "She sells seashells by the seashore.",
                "expected_difficulty": "medium",
                "target_accuracy": 0.85
            },
            {
                "category": "pronunciation",
                "text": "Peter Piper picked a peck of pickled peppers.",
                "expected_difficulty": "hard",
                "target_accuracy": 0.75
            },
            
            # Numbers and dates
            {
                "category": "numbers",
                "text": "The meeting is scheduled for January 15th, 2024 at 3:30 PM.",
                "expected_difficulty": "medium",
                "target_accuracy": 0.80
            },
            {
                "category": "numbers",
                "text": "The total cost is $1,234.56 including tax.",
                "expected_difficulty": "medium", 
                "target_accuracy": 0.80
            },
            
            # Technical terms
            {
                "category": "technical",
                "text": "The API endpoint returns JSON data with authentication tokens.",
                "expected_difficulty": "medium",
                "target_accuracy": 0.80
            },
            {
                "category": "technical",
                "text": "Machine learning algorithms process neural network architectures.",
                "expected_difficulty": "hard",
                "target_accuracy": 0.70
            },
            
            # Emotional content
            {
                "category": "emotion",
                "text": "I'm so excited about this amazing opportunity!",
                "expected_difficulty": "easy",
                "target_accuracy": 0.90,
                "emotion": "positive"
            },
            {
                "category": "emotion",
                "text": "This is a terrible and disappointing situation.",
                "expected_difficulty": "easy",
                "target_accuracy": 0.90,
                "emotion": "negative"
            },
            
            # Long form content
            {
                "category": "long_form",
                "text": "Artificial intelligence has revolutionized many industries by providing automated solutions to complex problems. Machine learning algorithms can now process vast amounts of data and identify patterns that would be impossible for humans to detect manually.",
                "expected_difficulty": "hard",
                "target_accuracy": 0.75
            },
            
            # Punctuation and formatting
            {
                "category": "punctuation",
                "text": "Hello! How are you today? I hope you're doing well... Let's continue.",
                "expected_difficulty": "medium",
                "target_accuracy": 0.85
            },
            
            # Mixed content
            {
                "category": "mixed",
                "text": "Dr. Smith's appointment is at 2:30 PM on March 3rd. Please call (555) 123-4567 to confirm.",
                "expected_difficulty": "hard",
                "target_accuracy": 0.70
            }
        ]
    
    async def run_full_validation(self, 
                                 voices: Optional[List[str]] = None,
                                 formats: Optional[List[str]] = None,
                                 speeds: Optional[List[float]] = None) -> Dict[str, Any]:
        """Run comprehensive validation across all test samples.
        
        Args:
            voices: List of voices to test (default: all available)
            formats: List of audio formats to test (default: ['mp3', 'wav'])
            speeds: List of speeds to test (default: [1.0])
            
        Returns:
            Comprehensive validation results
        """
        if voices is None:
            voices = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
        if formats is None:
            formats = ["mp3", "wav"]
        if speeds is None:
            speeds = [1.0]
        
        logger.info(f"Starting full validation: {len(self.test_samples)} samples, "
                   f"{len(voices)} voices, {len(formats)} formats, {len(speeds)} speeds")
        
        start_time = time.time()
        results = {
            "test_summary": {
                "total_tests": len(self.test_samples) * len(voices) * len(formats) * len(speeds),
                "start_time": start_time,
                "voices_tested": voices,
                "formats_tested": formats,
                "speeds_tested": speeds
            },
            "test_results": [],
            "category_summaries": {},
            "voice_summaries": {},
            "overall_summary": {}
        }
        
        test_count = 0
        passed_tests = 0
        failed_tests = 0
        
        # Run tests for each combination
        for sample in self.test_samples:
            for voice in voices:
                for format in formats:
                    for speed in speeds:
                        test_count += 1
                        logger.info(f"Running test {test_count}/{results['test_summary']['total_tests']}: "
                                  f"{sample['category']} - {voice} - {format} - {speed}x")
                        
                        try:
                            test_result = await self._run_single_test(sample, voice, format, speed)
                            results["test_results"].append(test_result)
                            
                            if test_result["passed"]:
                                passed_tests += 1
                            else:
                                failed_tests += 1
                                
                        except Exception as e:
                            logger.error(f"Test failed with exception: {e}")
                            failed_tests += 1
                            results["test_results"].append({
                                "sample": sample,
                                "voice": voice,
                                "format": format,
                                "speed": speed,
                                "passed": False,
                                "error": str(e),
                                "timestamp": time.time()
                            })
        
        # Calculate summaries
        end_time = time.time()
        results["test_summary"].update({
            "end_time": end_time,
            "duration": end_time - start_time,
            "total_tests_run": test_count,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / test_count if test_count > 0 else 0.0
        })
        
        # Generate category and voice summaries
        results["category_summaries"] = self._generate_category_summaries(results["test_results"])
        results["voice_summaries"] = self._generate_voice_summaries(results["test_results"])
        results["overall_summary"] = self._generate_overall_summary(results)
        
        logger.info(f"Validation completed: {passed_tests}/{test_count} tests passed "
                   f"({results['test_summary']['success_rate']:.1%} success rate)")
        
        return results
    
    async def _run_single_test(self, 
                              sample: Dict[str, Any], 
                              voice: str, 
                              format: str, 
                              speed: float) -> Dict[str, Any]:
        """Run a single validation test.
        
        Args:
            sample: Test sample configuration
            voice: Voice to use
            format: Audio format
            speed: Speech speed
            
        Returns:
            Single test result
        """
        test_start = time.time()
        
        try:
            # Generate TTS audio
            inference_engine = get_inference_engine()
            audio_processor = get_audio_processor()
            
            # Generate speech
            tts_result = await inference_engine.generate_speech(
                text=sample["text"],
                voice=voice,
                speed=speed,
                response_format=format
            )
            
            # Process audio
            audio_data = await audio_processor.process_audio(
                audio_array=tts_result["audio_data"],
                sample_rate=tts_result["sample_rate"],
                output_format=format,
                speed=speed
            )
            
            # Validate with Whisper
            validation_result = self.validator.validate_tts_output(
                sample["text"], 
                audio_data, 
                tts_result["sample_rate"]
            )
            
            if not validation_result["success"]:
                return {
                    "sample": sample,
                    "voice": voice,
                    "format": format,
                    "speed": speed,
                    "passed": False,
                    "error": validation_result.get("error", "Validation failed"),
                    "timestamp": test_start,
                    "duration": time.time() - test_start
                }
            
            # Assess quality
            quality_result = self.quality_assessor.assess_quality(
                sample["text"],
                validation_result["transcribed_text"],
                audio_data,
                validation_result
            )
            
            # Determine if test passed
            target_accuracy = sample.get("target_accuracy", 0.8)
            actual_accuracy = validation_result["accuracy_metrics"]["overall_accuracy"]
            overall_score = quality_result["overall_score"]
            
            passed = (actual_accuracy >= target_accuracy and 
                     quality_result["assessment_success"] and
                     overall_score >= 0.6)  # Minimum quality threshold
            
            return {
                "sample": sample,
                "voice": voice,
                "format": format,
                "speed": speed,
                "passed": passed,
                "validation_result": validation_result,
                "quality_result": quality_result,
                "target_accuracy": target_accuracy,
                "actual_accuracy": actual_accuracy,
                "overall_score": overall_score,
                "tts_metrics": {
                    "rtf": tts_result.get("rtf", 0),
                    "inference_time": tts_result.get("inference_time", 0),
                    "audio_duration": tts_result.get("audio_duration", 0)
                },
                "timestamp": test_start,
                "duration": time.time() - test_start
            }
            
        except Exception as e:
            logger.error(f"Single test failed: {e}")
            return {
                "sample": sample,
                "voice": voice,
                "format": format,
                "speed": speed,
                "passed": False,
                "error": str(e),
                "timestamp": test_start,
                "duration": time.time() - test_start
            }
    
    def _generate_category_summaries(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summaries by test category.
        
        Args:
            test_results: List of test results
            
        Returns:
            Category summaries
        """
        categories = {}
        
        for result in test_results:
            if "sample" not in result:
                continue
                
            category = result["sample"]["category"]
            if category not in categories:
                categories[category] = {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "avg_accuracy": 0.0,
                    "avg_quality_score": 0.0,
                    "accuracies": [],
                    "quality_scores": []
                }
            
            cat_data = categories[category]
            cat_data["total_tests"] += 1
            
            if result["passed"]:
                cat_data["passed_tests"] += 1
            else:
                cat_data["failed_tests"] += 1
            
            if "actual_accuracy" in result:
                cat_data["accuracies"].append(result["actual_accuracy"])
            if "overall_score" in result:
                cat_data["quality_scores"].append(result["overall_score"])
        
        # Calculate averages
        for category, data in categories.items():
            if data["accuracies"]:
                data["avg_accuracy"] = sum(data["accuracies"]) / len(data["accuracies"])
            if data["quality_scores"]:
                data["avg_quality_score"] = sum(data["quality_scores"]) / len(data["quality_scores"])
            data["success_rate"] = data["passed_tests"] / data["total_tests"] if data["total_tests"] > 0 else 0.0
            
            # Clean up temporary lists
            del data["accuracies"]
            del data["quality_scores"]
        
        return categories
    
    def _generate_voice_summaries(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summaries by voice.
        
        Args:
            test_results: List of test results
            
        Returns:
            Voice summaries
        """
        voices = {}
        
        for result in test_results:
            if "voice" not in result:
                continue
                
            voice = result["voice"]
            if voice not in voices:
                voices[voice] = {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "avg_accuracy": 0.0,
                    "avg_quality_score": 0.0,
                    "accuracies": [],
                    "quality_scores": []
                }
            
            voice_data = voices[voice]
            voice_data["total_tests"] += 1
            
            if result["passed"]:
                voice_data["passed_tests"] += 1
            else:
                voice_data["failed_tests"] += 1
            
            if "actual_accuracy" in result:
                voice_data["accuracies"].append(result["actual_accuracy"])
            if "overall_score" in result:
                voice_data["quality_scores"].append(result["overall_score"])
        
        # Calculate averages
        for voice, data in voices.items():
            if data["accuracies"]:
                data["avg_accuracy"] = sum(data["accuracies"]) / len(data["accuracies"])
            if data["quality_scores"]:
                data["avg_quality_score"] = sum(data["quality_scores"]) / len(data["quality_scores"])
            data["success_rate"] = data["passed_tests"] / data["total_tests"] if data["total_tests"] > 0 else 0.0
            
            # Clean up temporary lists
            del data["accuracies"]
            del data["quality_scores"]
        
        return voices
    
    def _generate_overall_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary.
        
        Args:
            results: Complete validation results
            
        Returns:
            Overall summary
        """
        test_summary = results["test_summary"]
        
        # Calculate average metrics
        all_accuracies = []
        all_quality_scores = []
        all_rtfs = []
        
        for result in results["test_results"]:
            if "actual_accuracy" in result:
                all_accuracies.append(result["actual_accuracy"])
            if "overall_score" in result:
                all_quality_scores.append(result["overall_score"])
            if "tts_metrics" in result and "rtf" in result["tts_metrics"]:
                all_rtfs.append(result["tts_metrics"]["rtf"])
        
        return {
            "success_rate": test_summary["success_rate"],
            "avg_accuracy": sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0,
            "avg_quality_score": sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0,
            "avg_rtf": sum(all_rtfs) / len(all_rtfs) if all_rtfs else 0.0,
            "total_duration": test_summary["duration"],
            "tests_per_minute": test_summary["total_tests_run"] / (test_summary["duration"] / 60) if test_summary["duration"] > 0 else 0.0,
            "performance_grade": self._calculate_performance_grade(test_summary["success_rate"]),
            "recommendations": self._generate_recommendations(results)
        }
    
    def _calculate_performance_grade(self, success_rate: float) -> str:
        """Calculate performance grade based on success rate.
        
        Args:
            success_rate: Overall success rate
            
        Returns:
            Performance grade
        """
        if success_rate >= 0.95:
            return "A"
        elif success_rate >= 0.85:
            return "B"
        elif success_rate >= 0.75:
            return "C"
        elif success_rate >= 0.65:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results.
        
        Args:
            results: Complete validation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        overall_summary = results.get("overall_summary", {})
        category_summaries = results.get("category_summaries", {})
        
        # Check overall performance
        if overall_summary.get("success_rate", 0) < 0.8:
            recommendations.append("Overall system performance needs improvement")
        
        if overall_summary.get("avg_rtf", 0) > 2.0:
            recommendations.append("Consider optimizing inference speed (RTF > 2.0)")
        
        # Check category-specific issues
        for category, data in category_summaries.items():
            if data.get("success_rate", 0) < 0.7:
                recommendations.append(f"Improve {category} handling (low success rate)")
            if data.get("avg_accuracy", 0) < 0.7:
                recommendations.append(f"Enhance {category} pronunciation accuracy")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations
    
    async def run_quick_validation(self, sample_count: int = 3) -> Dict[str, Any]:
        """Run a quick validation with a subset of tests.
        
        Args:
            sample_count: Number of samples to test
            
        Returns:
            Quick validation results
        """
        quick_samples = self.test_samples[:sample_count]
        voices = ["alloy"]  # Test with one voice only
        formats = ["mp3"]   # Test with one format only
        speeds = [1.0]      # Test with one speed only
        
        logger.info(f"Running quick validation with {len(quick_samples)} samples")
        
        # Temporarily replace test samples
        original_samples = self.test_samples
        self.test_samples = quick_samples
        
        try:
            results = await self.run_full_validation(voices, formats, speeds)
            results["validation_type"] = "quick"
            return results
        finally:
            self.test_samples = original_samples
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save validation results to file.
        
        Args:
            results: Validation results
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Global test suite instance
_test_suite: Optional[ValidationTestSuite] = None


def get_validation_test_suite(whisper_model_size: str = "base") -> ValidationTestSuite:
    """Get the global validation test suite instance.
    
    Args:
        whisper_model_size: Whisper model size to use
        
    Returns:
        Global ValidationTestSuite instance
    """
    global _test_suite
    if _test_suite is None or _test_suite.whisper_model_size != whisper_model_size:
        _test_suite = ValidationTestSuite(whisper_model_size)
    return _test_suite
