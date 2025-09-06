#!/usr/bin/env python3
"""
Final Performance Validation for JabberTTS
==========================================

This script validates that we have achieved the RTF ≤ 0.25 target through our optimizations:

1. Tests the ultra-optimized SpeechT5 implementation
2. Integrates optimizations into the main inference engine
3. Validates performance across all test cases
4. Provides final performance report

Target: RTF ≤ 0.25 for all test cases
"""

import asyncio
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# JabberTTS imports
from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.models.manager import ModelManager

# Import our ultra-optimized model
from ultra_optimized_speecht5 import UltraOptimizedSpeechT5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalPerformanceValidator:
    """Validates final performance after all optimizations."""

    def __init__(self):
        """Initialize the performance validator."""
        self.inference_engine = None
        self.audio_processor = None
        self.model_manager = None

        # Test configurations
        self.test_cases = [
            {"name": "short", "text": "Hello world", "expected_duration": 0.7},
            {"name": "medium", "text": "This is a medium length sentence for testing performance.", "expected_duration": 2.4},
            {"name": "long", "text": "This is a much longer sentence that contains multiple clauses and should take significantly more time to process, allowing us to measure how performance scales with text length and complexity.", "expected_duration": 7.8},
            {"name": "complex", "text": "The quick brown fox jumps over the lazy dog. Numbers: 123, 456, 789. Punctuation: Hello, world! How are you? I'm fine, thanks.", "expected_duration": 5.2},
            {"name": "very_short", "text": "Hi", "expected_duration": 0.3},
            {"name": "numbers", "text": "The year 2024 has 365 days and 12 months.", "expected_duration": 3.1}
        ]

        # Results storage
        self.validation_results = {
            "timestamp": time.time(),
            "optimizations_applied": [],
            "test_results": {},
            "performance_summary": {},
            "success": False
        }

    async def run_final_validation(self) -> Dict[str, Any]:
        """Run the final performance validation."""
        logger.info("🎯 FINAL PERFORMANCE VALIDATION")
        logger.info("=" * 60)
        logger.info("Target: RTF ≤ 0.25 for all test cases")
        logger.info("=" * 60)

        try:
            # Step 1: Apply ultra-optimizations to the system
            await self._apply_system_optimizations()

            # Step 2: Test with ultra-optimized model directly
            await self._test_ultra_optimized_model()

            # Step 3: Test through the full inference engine
            await self._test_full_inference_engine()

            # Step 4: Validate performance targets
            await self._validate_performance_targets()

            # Step 5: Generate final report
            await self._generate_final_report()

            return self.validation_results

        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            raise

    async def _apply_system_optimizations(self):
        """Apply system-wide optimizations."""
        logger.info("🚀 Applying System-Wide Optimizations")

        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        os.environ['NUMEXPR_NUM_THREADS'] = '2'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # Force SpeechT5 model
        os.environ['JABBERTTS_MODEL_NAME'] = 'speecht5'

        self.validation_results["optimizations_applied"].extend([
            "System threading optimization",
            "Memory allocation optimization",
            "Model selection optimization"
        ])

        logger.info("✅ System optimizations applied")

    async def _test_ultra_optimized_model(self):
        """Test the ultra-optimized model directly."""
        logger.info("🧪 Testing Ultra-Optimized Model Directly")

        # Create and load ultra-optimized model
        model = UltraOptimizedSpeechT5(Path("."), "cpu")
        model.load_model()

        direct_results = {}

        for test_case in self.test_cases:
            logger.info(f"  Testing: {test_case['name']}")

            # Run test with timing
            start_time = time.time()
            audio = model.generate_speech(
                text=test_case["text"],
                voice="alloy",
                speed=1.0
            )
            total_time = time.time() - start_time

            # Calculate RTF
            audio_duration = len(audio) / model.SAMPLE_RATE
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

            direct_results[test_case["name"]] = {
                "total_time": total_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "target_met": rtf <= 0.25,
                "audio_samples": len(audio)
            }

            logger.info(f"    RTF: {rtf:.3f} ({'✅' if rtf <= 0.25 else '❌'})")

        self.validation_results["test_results"]["direct_model"] = direct_results

        # Get optimization stats
        stats = model.get_optimization_stats()
        self.validation_results["optimization_stats"] = stats

        logger.info(f"✅ Direct model testing completed")

    async def _test_full_inference_engine(self):
        """Test through the full inference engine."""
        logger.info("🔧 Testing Full Inference Engine")

        # Initialize inference engine
        self.inference_engine = InferenceEngine()
        self.audio_processor = AudioProcessor()

        engine_results = {}

        for test_case in self.test_cases:
            logger.info(f"  Testing: {test_case['name']}")

            # Run test through full engine
            start_time = time.time()
            result = await self.inference_engine.generate_speech(
                text=test_case["text"],
                voice="alloy",
                response_format="wav"
            )
            total_time = time.time() - start_time

            # Calculate RTF
            audio_duration = len(result["audio_data"]) / result["sample_rate"]
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

            engine_results[test_case["name"]] = {
                "total_time": total_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "target_met": rtf <= 0.25,
                "sample_rate": result["sample_rate"],
                "audio_samples": len(result["audio_data"])
            }

            logger.info(f"    RTF: {rtf:.3f} ({'✅' if rtf <= 0.25 else '❌'})")

        self.validation_results["test_results"]["full_engine"] = engine_results

        logger.info(f"✅ Full engine testing completed")

    async def _validate_performance_targets(self):
        """Validate that performance targets are met."""
        logger.info("🎯 Validating Performance Targets")

        # Check direct model results
        direct_results = self.validation_results["test_results"]["direct_model"]
        direct_passed = sum(1 for data in direct_results.values() if data["target_met"])
        direct_total = len(direct_results)

        # Check full engine results
        engine_results = self.validation_results["test_results"]["full_engine"]
        engine_passed = sum(1 for data in engine_results.values() if data["target_met"])
        engine_total = len(engine_results)

        # Calculate averages
        direct_avg_rtf = sum(data["rtf"] for data in direct_results.values()) / direct_total
        engine_avg_rtf = sum(data["rtf"] for data in engine_results.values()) / engine_total

        # Performance summary
        performance_summary = {
            "direct_model": {
                "tests_passed": direct_passed,
                "total_tests": direct_total,
                "success_rate": (direct_passed / direct_total) * 100,
                "average_rtf": direct_avg_rtf,
                "target_achieved": direct_passed == direct_total
            },
            "full_engine": {
                "tests_passed": engine_passed,
                "total_tests": engine_total,
                "success_rate": (engine_passed / engine_total) * 100,
                "average_rtf": engine_avg_rtf,
                "target_achieved": engine_passed == engine_total
            }
        }

        self.validation_results["performance_summary"] = performance_summary

        # Overall success
        overall_success = (direct_passed == direct_total) and (engine_passed == engine_total)
        self.validation_results["success"] = overall_success

        logger.info(f"📊 Performance Summary:")
        logger.info(f"  Direct Model: {direct_passed}/{direct_total} tests passed ({(direct_passed/direct_total)*100:.1f}%)")
        logger.info(f"  Full Engine: {engine_passed}/{engine_total} tests passed ({(engine_passed/engine_total)*100:.1f}%)")
        logger.info(f"  Overall Success: {'✅ YES' if overall_success else '❌ NO'}")

    async def _generate_final_report(self):
        """Generate the final performance report."""
        logger.info("📄 Generating Final Report")

        # Save detailed results
        output_file = Path("final_performance_validation.json")
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        # Generate summary report
        summary_file = Path("PERFORMANCE_VALIDATION_SUMMARY.md")
        await self._generate_summary_report(summary_file)

        logger.info(f"📄 Detailed results saved to: {output_file}")
        logger.info(f"📄 Summary report saved to: {summary_file}")

    async def _generate_summary_report(self, output_file: Path):
        """Generate a human-readable summary report."""
        performance = self.validation_results["performance_summary"]

        summary = f"""# JabberTTS Performance Validation Summary
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 PERFORMANCE TARGET: RTF ≤ 0.25

## ✅ VALIDATION RESULTS

### Direct Model Performance
- **Tests Passed**: {performance['direct_model']['tests_passed']}/{performance['direct_model']['total_tests']}
- **Success Rate**: {performance['direct_model']['success_rate']:.1f}%
- **Average RTF**: {performance['direct_model']['average_rtf']:.3f}
- **Target Achieved**: {'✅ YES' if performance['direct_model']['target_achieved'] else '❌ NO'}

### Full Engine Performance
- **Tests Passed**: {performance['full_engine']['tests_passed']}/{performance['full_engine']['total_tests']}
- **Success Rate**: {performance['full_engine']['success_rate']:.1f}%
- **Average RTF**: {performance['full_engine']['average_rtf']:.3f}
- **Target Achieved**: {'✅ YES' if performance['full_engine']['target_achieved'] else '❌ NO'}

## 📊 DETAILED TEST RESULTS

### Direct Model Results
"""

        direct_results = self.validation_results["test_results"]["direct_model"]
        for test_name, data in direct_results.items():
            status = "✅ PASS" if data["target_met"] else "❌ FAIL"
            summary += f"- **{test_name.title()}**: RTF {data['rtf']:.3f} {status}\n"

        summary += f"""
### Full Engine Results
"""

        engine_results = self.validation_results["test_results"]["full_engine"]
        for test_name, data in engine_results.items():
            status = "✅ PASS" if data["target_met"] else "❌ FAIL"
            summary += f"- **{test_name.title()}**: RTF {data['rtf']:.3f} {status}\n"

        summary += f"""
## 🚀 OPTIMIZATIONS APPLIED
"""
        for optimization in self.validation_results["optimizations_applied"]:
            summary += f"- {optimization}\n"

        summary += f"""
## 🏆 FINAL VERDICT
**Overall Success**: {'✅ TARGET ACHIEVED' if self.validation_results['success'] else '❌ TARGET NOT ACHIEVED'}

The JabberTTS system {'has successfully achieved' if self.validation_results['success'] else 'has not yet achieved'} the performance target of RTF ≤ 0.25 across all test cases.
"""

        with open(output_file, 'w') as f:
            f.write(summary)


async def main():
    """Run the final performance validation."""
    validator = FinalPerformanceValidator()

    print("🎯 JABBERTTS FINAL PERFORMANCE VALIDATION")
    print("=" * 60)

    try:
        results = await validator.run_final_validation()

        print("\n🏆 VALIDATION COMPLETED")
        print("=" * 60)

        success = results["success"]
        performance = results["performance_summary"]

        print(f"🎯 TARGET: RTF ≤ 0.25")
        print(f"✅ SUCCESS: {'YES' if success else 'NO'}")
        print(f"📊 DIRECT MODEL: {performance['direct_model']['success_rate']:.1f}% success rate")
        print(f"📊 FULL ENGINE: {performance['full_engine']['success_rate']:.1f}% success rate")

        if success:
            print("\n🎉 CONGRATULATIONS! RTF ≤ 0.25 TARGET ACHIEVED!")
        else:
            print("\n⚠️  RTF target not fully achieved. Further optimization needed.")

        return results

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())