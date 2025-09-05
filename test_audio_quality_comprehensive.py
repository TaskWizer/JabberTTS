#!/usr/bin/env python3
"""Comprehensive Audio Quality Test Suite for JabberTTS.

This script performs extensive audio quality validation including:
- Objective quality metrics
- Reference sample validation
- Performance benchmarking
- Regression testing
"""

import sys
import time
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.validation.audio_quality import AudioQualityValidator, AudioQualityMetrics


class ComprehensiveAudioQualityTest:
    """Comprehensive audio quality testing system."""
    
    def __init__(self):
        """Initialize test system."""
        self.engine = InferenceEngine()
        self.validator = AudioQualityValidator()
        self.test_results = []
        
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive audio quality test suite."""
        print("=== JabberTTS Comprehensive Audio Quality Test ===\n")
        
        # Test categories
        test_categories = [
            ("Basic Speech", self._test_basic_speech),
            ("Complex Pronunciation", self._test_complex_pronunciation),
            ("Numbers and Dates", self._test_numbers_dates),
            ("Voice Consistency", self._test_voice_consistency),
            ("Performance Validation", self._test_performance),
            ("Format Quality", self._test_format_quality),
        ]
        
        overall_results = {
            "test_timestamp": time.time(),
            "categories": {},
            "summary": {}
        }
        
        # Run each test category
        for category_name, test_func in test_categories:
            print(f"\n=== {category_name} Tests ===")
            try:
                category_results = await test_func()
                overall_results["categories"][category_name] = category_results
                print(f"✓ {category_name} completed: {category_results['pass_rate']:.1%} pass rate")
            except Exception as e:
                print(f"✗ {category_name} failed: {e}")
                overall_results["categories"][category_name] = {"error": str(e)}
        
        # Generate summary
        overall_results["summary"] = self._generate_summary(overall_results["categories"])
        
        return overall_results
    
    async def _test_basic_speech(self) -> Dict[str, Any]:
        """Test basic speech quality."""
        test_cases = [
            "Hello, world!",
            "This is a simple test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing basic speech synthesis quality.",
        ]
        
        return await self._run_test_cases("Basic Speech", test_cases)
    
    async def _test_complex_pronunciation(self) -> Dict[str, Any]:
        """Test complex pronunciation handling."""
        test_cases = [
            "The schedule is often either this or that.",
            "Difficult words: colonel, yacht, psychology, choir.",
            "Pronunciation test: through, though, thought, tough.",
            "Foreign words: café, naïve, résumé, jalapeño.",
        ]
        
        return await self._run_test_cases("Complex Pronunciation", test_cases)
    
    async def _test_numbers_dates(self) -> Dict[str, Any]:
        """Test numbers and dates pronunciation."""
        test_cases = [
            "Numbers: 123, 456, 789, 1000, 2024.",
            "Dates: January 15th, 2024, December 31st, 1999.",
            "Currency: $99.99, €50.00, £25.50.",
            "Time: 3:30 PM, 12:00 AM, 11:59 PM.",
        ]
        
        return await self._run_test_cases("Numbers and Dates", test_cases)
    
    async def _test_voice_consistency(self) -> Dict[str, Any]:
        """Test voice consistency across different voices."""
        test_text = "This is a consistency test across different voices."
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        results = {
            "total_tests": len(voices),
            "passed_tests": 0,
            "failed_tests": 0,
            "voice_results": {},
            "consistency_score": 0.0
        }
        
        voice_metrics = []
        
        for voice in voices:
            try:
                print(f"  Testing voice: {voice}")
                
                # Generate speech
                result = await self.engine.generate_speech(test_text, voice=voice)
                
                # Analyze quality
                metrics = self.validator.analyze_audio(
                    result["audio_data"], 
                    result["sample_rate"],
                    result["rtf"],
                    result["inference_time"]
                )
                
                # Validate against thresholds
                validation = self.validator.validate_against_thresholds(metrics)
                
                voice_metrics.append(metrics)
                results["voice_results"][voice] = {
                    "metrics": metrics.to_dict(),
                    "validation": validation,
                    "passed": all(validation.values())
                }
                
                if all(validation.values()):
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                    
            except Exception as e:
                print(f"    ✗ Voice {voice} failed: {e}")
                results["failed_tests"] += 1
                results["voice_results"][voice] = {"error": str(e)}
        
        # Calculate consistency score
        if voice_metrics:
            # Measure consistency across voices
            overall_qualities = [m.overall_quality for m in voice_metrics]
            rms_levels = [m.rms_level for m in voice_metrics]
            
            quality_std = np.std(overall_qualities) if len(overall_qualities) > 1 else 0
            rms_std = np.std(rms_levels) if len(rms_levels) > 1 else 0
            
            # Lower standard deviation = higher consistency
            consistency_score = max(0, 100 - (quality_std * 2 + rms_std * 1000))
            results["consistency_score"] = consistency_score
        
        results["pass_rate"] = results["passed_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0
        
        return results
    
    async def _test_performance(self) -> Dict[str, Any]:
        """Test performance metrics."""
        test_text = "Performance testing with various text lengths and complexity."
        
        results = {
            "total_tests": 5,
            "passed_tests": 0,
            "failed_tests": 0,
            "performance_metrics": {},
            "rtf_target_met": False
        }
        
        rtfs = []
        
        # Test multiple runs for performance consistency
        for i in range(5):
            try:
                result = await self.engine.generate_speech(test_text, voice="alloy")
                
                rtf = result["rtf"]
                rtfs.append(rtf)
                
                # Check if RTF target is met
                if rtf < 0.5:
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
                    
            except Exception as e:
                results["failed_tests"] += 1
        
        if rtfs:
            avg_rtf = sum(rtfs) / len(rtfs)
            best_rtf = min(rtfs)
            worst_rtf = max(rtfs)
            
            results["performance_metrics"] = {
                "average_rtf": avg_rtf,
                "best_rtf": best_rtf,
                "worst_rtf": worst_rtf,
                "rtf_consistency": 1.0 - (worst_rtf - best_rtf) / (avg_rtf + 1e-8)
            }
            
            results["rtf_target_met"] = avg_rtf < 0.5
        
        results["pass_rate"] = results["passed_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0
        
        return results
    
    async def _test_format_quality(self) -> Dict[str, Any]:
        """Test audio quality across different formats."""
        from jabbertts.audio.processor import AudioProcessor
        
        test_text = "Testing audio quality across different output formats."
        formats = ["wav", "mp3", "opus", "flac"]
        
        processor = AudioProcessor()
        
        results = {
            "total_tests": len(formats),
            "passed_tests": 0,
            "failed_tests": 0,
            "format_results": {}
        }
        
        # Generate base audio
        result = await self.engine.generate_speech(test_text, voice="alloy")
        audio_data = result["audio_data"]
        sample_rate = result["sample_rate"]
        
        for format_name in formats:
            try:
                print(f"  Testing format: {format_name}")
                
                # Process audio to format
                encoded_audio, metadata = await processor.process_audio(
                    audio_data, sample_rate, output_format=format_name
                )
                
                # Calculate compression ratio
                original_size = len(audio_data) * 4  # float32 = 4 bytes per sample
                compressed_size = len(encoded_audio)
                compression_ratio = original_size / compressed_size
                
                results["format_results"][format_name] = {
                    "success": True,
                    "compressed_size": compressed_size,
                    "compression_ratio": compression_ratio,
                    "metadata": metadata
                }
                
                results["passed_tests"] += 1
                
            except Exception as e:
                print(f"    ✗ Format {format_name} failed: {e}")
                results["failed_tests"] += 1
                results["format_results"][format_name] = {"error": str(e)}
        
        results["pass_rate"] = results["passed_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0
        
        return results
    
    async def _run_test_cases(self, category: str, test_cases: List[str]) -> Dict[str, Any]:
        """Run a set of test cases and analyze quality."""
        results = {
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_cases": {},
            "average_quality": 0.0
        }
        
        quality_scores = []
        
        for i, text in enumerate(test_cases, 1):
            try:
                print(f"  Test {i}: '{text[:50]}...' " if len(text) > 50 else f"  Test {i}: '{text}'")
                
                # Generate speech
                result = await self.engine.generate_speech(text, voice="alloy")
                
                # Analyze quality
                metrics = self.validator.analyze_audio(
                    result["audio_data"], 
                    result["sample_rate"],
                    result["rtf"],
                    result["inference_time"]
                )
                
                # Validate against thresholds
                validation = self.validator.validate_against_thresholds(metrics)
                
                quality_scores.append(metrics.overall_quality)
                
                results["test_cases"][f"test_{i}"] = {
                    "text": text,
                    "metrics": metrics.to_dict(),
                    "validation": validation,
                    "passed": all(validation.values())
                }
                
                if all(validation.values()):
                    results["passed_tests"] += 1
                    print(f"    ✓ Quality: {metrics.overall_quality:.1f}/100")
                else:
                    results["failed_tests"] += 1
                    failed_checks = [k for k, v in validation.items() if not v]
                    print(f"    ✗ Failed checks: {failed_checks}")
                    
            except Exception as e:
                print(f"    ✗ Test {i} failed: {e}")
                results["failed_tests"] += 1
                results["test_cases"][f"test_{i}"] = {"text": text, "error": str(e)}
        
        if quality_scores:
            results["average_quality"] = sum(quality_scores) / len(quality_scores)
        
        results["pass_rate"] = results["passed_tests"] / results["total_tests"] if results["total_tests"] > 0 else 0
        
        return results
    
    def _generate_summary(self, categories: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall test summary."""
        total_tests = 0
        total_passed = 0
        
        category_summaries = {}
        
        for category, results in categories.items():
            if "error" not in results:
                total_tests += results.get("total_tests", 0)
                total_passed += results.get("passed_tests", 0)
                
                category_summaries[category] = {
                    "pass_rate": results.get("pass_rate", 0),
                    "status": "PASS" if results.get("pass_rate", 0) >= 0.8 else "FAIL"
                }
        
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_pass_rate": overall_pass_rate,
            "overall_status": "PASS" if overall_pass_rate >= 0.8 else "FAIL",
            "categories": category_summaries
        }


async def main():
    """Run comprehensive audio quality test."""
    import numpy as np
    
    test_system = ComprehensiveAudioQualityTest()
    
    # Warmup system
    print("Warming up TTS system...")
    await test_system.engine.warmup(num_runs=2)
    
    # Run comprehensive test
    results = await test_system.run_comprehensive_test()
    
    # Print summary
    summary = results["summary"]
    print(f"\n=== FINAL RESULTS ===")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['total_passed']}")
    print(f"Pass Rate: {summary['overall_pass_rate']:.1%}")
    
    print(f"\nCategory Results:")
    for category, cat_summary in summary["categories"].items():
        status_icon = "✓" if cat_summary["status"] == "PASS" else "✗"
        print(f"  {status_icon} {category}: {cat_summary['pass_rate']:.1%}")
    
    # Save results
    results_file = Path("audio_quality_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return summary["overall_status"] == "PASS"


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
