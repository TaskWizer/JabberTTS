#!/usr/bin/env python3
"""
Comprehensive Performance Audit for JabberTTS
==============================================

This script conducts a deep technical audit of the JabberTTS audio generation pipeline
to identify fundamental performance and quality issues causing RTF ~0.5 instead of target ≤0.25.

Key Analysis Areas:
1. Pipeline Stage Timing Analysis
2. Memory Usage Profiling
3. Model Loading & Caching Efficiency
4. Hardware Utilization Assessment
5. Bottleneck Identification & Quantification
"""

import asyncio
import cProfile
import io
import json
import logging
import pstats
import time
import tracemalloc
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil
import numpy as np

# JabberTTS imports
from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.models.manager import ModelManager
from jabbertts.metrics import MetricsCollector
from jabbertts.optimization.performance_enhancer import PerformanceEnhancer
from jabbertts.caching.multilevel_cache import MultiLevelCacheManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceAuditor:
    """Comprehensive performance auditor for JabberTTS pipeline."""

    def __init__(self, output_dir: str = "performance_audit_results"):
        """Initialize the performance auditor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.inference_engine = None
        self.audio_processor = None
        self.model_manager = None
        self.metrics_collector = None
        self.performance_enhancer = None

        # Test configurations
        self.test_texts = [
            {"name": "short", "text": "Hello world", "expected_duration": 1.0},
            {"name": "medium", "text": "This is a medium length sentence for testing performance.", "expected_duration": 3.0},
            {"name": "long", "text": "This is a much longer sentence that contains multiple clauses and should take significantly more time to process, allowing us to measure how performance scales with text length and complexity.", "expected_duration": 8.0},
            {"name": "complex", "text": "The quick brown fox jumps over the lazy dog. Numbers: 123, 456, 789. Punctuation: Hello, world! How are you? I'm fine, thanks.", "expected_duration": 6.0}
        ]

        # Results storage
        self.audit_results = {
            "audit_timestamp": datetime.now().isoformat(),
            "system_info": {},
            "pipeline_analysis": {},
            "performance_bottlenecks": {},
            "memory_analysis": {},
            "optimization_effectiveness": {},
            "recommendations": []
        }

    async def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run the complete performance audit."""
        logger.info("🔍 Starting Comprehensive Performance Audit")
        logger.info("=" * 60)

        try:
            # Phase 1: System Information Collection
            await self._collect_system_info()

            # Phase 2: Component Initialization Analysis
            await self._analyze_component_initialization()

            # Phase 3: Pipeline Stage Timing Analysis
            await self._analyze_pipeline_stages()

            # Phase 4: Memory Usage Profiling
            await self._analyze_memory_usage()

            # Phase 5: Caching Effectiveness Analysis
            await self._analyze_caching_effectiveness()

            # Phase 6: Hardware Utilization Assessment
            await self._analyze_hardware_utilization()

            # Phase 7: Bottleneck Identification
            await self._identify_performance_bottlenecks()

            # Phase 8: Generate Recommendations
            await self._generate_recommendations()

            # Save results
            await self._save_audit_results()

            return self.audit_results

        except Exception as e:
            logger.error(f"Audit failed: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _collect_system_info(self):
        """Collect system information and hardware details."""
        logger.info("📊 Collecting System Information")

        # CPU information
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "cpu_percent": psutil.cpu_percent(interval=1)
        }

        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }

        # GPU information (if available)
        gpu_info = {"available": False}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_allocated": torch.cuda.memory_allocated() / (1024**3),
                    "memory_reserved": torch.cuda.memory_reserved() / (1024**3)
                }
        except ImportError:
            pass

        self.audit_results["system_info"] = {
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "python_version": psutil.__version__
        }

        logger.info(f"CPU: {cpu_info['cpu_count']} cores, {cpu_info['cpu_count_logical']} logical")
        logger.info(f"Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available")
        logger.info(f"GPU: {'Available' if gpu_info['available'] else 'Not available'}")

    async def _analyze_component_initialization(self):
        """Analyze component initialization times and overhead."""
        logger.info("🚀 Analyzing Component Initialization")

        initialization_times = {}

        # Time inference engine initialization
        start_time = time.time()
        self.inference_engine = InferenceEngine()
        initialization_times["inference_engine"] = time.time() - start_time

        # Time audio processor initialization
        start_time = time.time()
        self.audio_processor = AudioProcessor()
        initialization_times["audio_processor"] = time.time() - start_time

        # Time model manager initialization
        start_time = time.time()
        self.model_manager = ModelManager()
        initialization_times["model_manager"] = time.time() - start_time

        # Time model loading (cold start)
        start_time = time.time()
        model = await self.inference_engine._ensure_model_loaded()
        model_load_time = time.time() - start_time
        initialization_times["model_loading"] = model_load_time

        # Time warm start (model already loaded)
        start_time = time.time()
        model = await self.inference_engine._ensure_model_loaded()
        warm_start_time = time.time() - start_time
        initialization_times["model_warm_start"] = warm_start_time

        self.audit_results["pipeline_analysis"]["initialization"] = {
            "times": initialization_times,
            "total_cold_start": sum(initialization_times.values()),
            "model_info": {
                "class": type(model).__name__,
                "sample_rate": model.get_sample_rate(),
                "is_loaded": model.is_loaded
            }
        }

        logger.info(f"Total cold start time: {sum(initialization_times.values()):.3f}s")
        logger.info(f"Model loading: {model_load_time:.3f}s")
        logger.info(f"Model warm start: {warm_start_time:.3f}s")

    async def _analyze_pipeline_stages(self):
        """Analyze timing for each stage of the TTS pipeline."""
        logger.info("⏱️  Analyzing Pipeline Stage Timing")

        stage_analysis = {}

        for test_config in self.test_texts:
            logger.info(f"Testing: {test_config['name']} - '{test_config['text'][:50]}...'")

            # Start memory tracking
            tracemalloc.start()

            # Stage timing analysis
            stage_times = {}
            total_start = time.time()

            # Stage 1: Text preprocessing
            preprocess_start = time.time()
            model = await self.inference_engine._ensure_model_loaded()
            processed_text = await self.inference_engine._preprocess_text(test_config["text"], model)
            stage_times["preprocessing"] = time.time() - preprocess_start

            # Stage 2: Model inference
            inference_start = time.time()
            audio_data = await self.inference_engine._generate_audio(model, processed_text, "alloy", 1.0)
            stage_times["model_inference"] = time.time() - inference_start

            # Stage 3: Audio processing
            processing_start = time.time()
            audio_bytes, metadata = await self.audio_processor.process_audio(
                audio_data, model.get_sample_rate(), "wav"
            )
            stage_times["audio_processing"] = time.time() - processing_start

            total_time = time.time() - total_start
            stage_times["total"] = total_time

            # Calculate RTF
            audio_duration = len(audio_data) / model.get_sample_rate()
            rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

            # Memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            stage_analysis[test_config["name"]] = {
                "text_length": len(test_config["text"]),
                "audio_duration": audio_duration,
                "stage_times": stage_times,
                "rtf": rtf,
                "memory_usage": {
                    "current_mb": current / (1024**2),
                    "peak_mb": peak / (1024**2)
                },
                "performance_target_met": rtf <= 0.25
            }

            logger.info(f"  RTF: {rtf:.3f} ({'✓' if rtf <= 0.25 else '✗'})")
            logger.info(f"  Breakdown: Preprocess={stage_times['preprocessing']:.3f}s, "
                       f"Inference={stage_times['model_inference']:.3f}s, "
                       f"Processing={stage_times['audio_processing']:.3f}s")

        self.audit_results["pipeline_analysis"]["stage_timing"] = stage_analysis

    async def _analyze_memory_usage(self):
        """Analyze memory usage patterns and identify memory leaks."""
        logger.info("🧠 Analyzing Memory Usage Patterns")

        memory_analysis = {}

        # Baseline memory usage
        baseline_memory = psutil.virtual_memory().used / (1024**2)

        # Test memory usage with different text lengths
        for test_config in self.test_texts:
            logger.info(f"Memory test: {test_config['name']}")

            # Memory before generation
            memory_before = psutil.virtual_memory().used / (1024**2)

            # Generate audio multiple times to check for leaks
            for i in range(3):
                result = await self.inference_engine.generate_speech(
                    text=test_config["text"],
                    voice="alloy",
                    response_format="wav"
                )

                # Process audio
                audio_bytes, metadata = await self.audio_processor.process_audio(
                    result["audio_data"], result["sample_rate"], "wav"
                )

            # Memory after generation
            memory_after = psutil.virtual_memory().used / (1024**2)
            memory_delta = memory_after - memory_before

            memory_analysis[test_config["name"]] = {
                "memory_before_mb": memory_before,
                "memory_after_mb": memory_after,
                "memory_delta_mb": memory_delta,
                "potential_leak": memory_delta > 50  # Flag if >50MB increase
            }

            logger.info(f"  Memory delta: {memory_delta:.1f}MB")

        self.audit_results["memory_analysis"] = memory_analysis

    async def _analyze_caching_effectiveness(self):
        """Analyze caching system effectiveness."""
        logger.info("💾 Analyzing Caching Effectiveness")

        # Test cache performance with repeated requests
        cache_test_text = "This is a test sentence for cache analysis."

        # First request (cache miss)
        start_time = time.time()
        result1 = await self.inference_engine.generate_speech(
            text=cache_test_text,
            voice="alloy",
            response_format="wav"
        )
        first_request_time = time.time() - start_time

        # Second request (potential cache hit)
        start_time = time.time()
        result2 = await self.inference_engine.generate_speech(
            text=cache_test_text,
            voice="alloy",
            response_format="wav"
        )
        second_request_time = time.time() - start_time

        # Calculate cache effectiveness
        cache_speedup = first_request_time / second_request_time if second_request_time > 0 else 1.0

        self.audit_results["optimization_effectiveness"]["caching"] = {
            "first_request_time": first_request_time,
            "second_request_time": second_request_time,
            "cache_speedup": cache_speedup,
            "cache_effective": cache_speedup > 1.2  # 20% improvement threshold
        }

        logger.info(f"Cache speedup: {cache_speedup:.2f}x")

    async def _analyze_hardware_utilization(self):
        """Analyze hardware utilization during TTS generation."""
        logger.info("⚡ Analyzing Hardware Utilization")

        # Monitor CPU and memory during generation
        cpu_usage = []
        memory_usage = []

        async def monitor_resources():
            for _ in range(10):  # Monitor for 10 seconds
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
                memory_usage.append(psutil.virtual_memory().percent)
                await asyncio.sleep(0.1)

        # Start monitoring
        monitor_task = asyncio.create_task(monitor_resources())

        # Generate audio during monitoring
        test_text = "This is a comprehensive test sentence for hardware utilization analysis."
        result = await self.inference_engine.generate_speech(
            text=test_text,
            voice="alloy",
            response_format="wav"
        )

        # Wait for monitoring to complete
        await monitor_task

        # Analyze utilization
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        max_cpu = max(cpu_usage)
        avg_memory = sum(memory_usage) / len(memory_usage)
        max_memory = max(memory_usage)

        self.audit_results["optimization_effectiveness"]["hardware_utilization"] = {
            "cpu_usage": {
                "average": avg_cpu,
                "maximum": max_cpu,
                "underutilized": avg_cpu < 50  # Flag if <50% average usage
            },
            "memory_usage": {
                "average": avg_memory,
                "maximum": max_memory
            }
        }

        logger.info(f"CPU utilization: {avg_cpu:.1f}% avg, {max_cpu:.1f}% max")
        logger.info(f"Memory utilization: {avg_memory:.1f}% avg, {max_memory:.1f}% max")

    async def _identify_performance_bottlenecks(self):
        """Identify the top performance bottlenecks."""
        logger.info("🎯 Identifying Performance Bottlenecks")

        bottlenecks = []
        stage_timing = self.audit_results["pipeline_analysis"]["stage_timing"]

        # Analyze each test case to identify bottlenecks
        for test_name, data in stage_timing.items():
            stage_times = data["stage_times"]
            total_time = stage_times["total"]

            # Calculate percentage of time spent in each stage
            stage_percentages = {
                stage: (time_spent / total_time) * 100
                for stage, time_spent in stage_times.items()
                if stage != "total"
            }

            # Identify the biggest bottleneck for this test
            biggest_bottleneck = max(stage_percentages.items(), key=lambda x: x[1])

            bottlenecks.append({
                "test_case": test_name,
                "rtf": data["rtf"],
                "target_met": data["performance_target_met"],
                "biggest_bottleneck": {
                    "stage": biggest_bottleneck[0],
                    "percentage": biggest_bottleneck[1],
                    "time_seconds": stage_times[biggest_bottleneck[0]]
                },
                "stage_breakdown": stage_percentages
            })

        # Overall bottleneck analysis
        all_stage_times = {}
        for test_name, data in stage_timing.items():
            for stage, time_spent in data["stage_times"].items():
                if stage != "total":
                    if stage not in all_stage_times:
                        all_stage_times[stage] = []
                    all_stage_times[stage].append(time_spent)

        # Calculate average time per stage
        avg_stage_times = {
            stage: sum(times) / len(times)
            for stage, times in all_stage_times.items()
        }

        # Identify overall bottleneck
        overall_bottleneck = max(avg_stage_times.items(), key=lambda x: x[1])

        self.audit_results["performance_bottlenecks"] = {
            "per_test_case": bottlenecks,
            "overall_bottleneck": {
                "stage": overall_bottleneck[0],
                "average_time": overall_bottleneck[1]
            },
            "average_stage_times": avg_stage_times
        }

        logger.info(f"Overall bottleneck: {overall_bottleneck[0]} ({overall_bottleneck[1]:.3f}s avg)")

    async def _generate_recommendations(self):
        """Generate optimization recommendations based on analysis."""
        logger.info("💡 Generating Optimization Recommendations")

        recommendations = []

        # Analyze RTF performance
        stage_timing = self.audit_results["pipeline_analysis"]["stage_timing"]
        failed_tests = [name for name, data in stage_timing.items() if not data["performance_target_met"]]

        if failed_tests:
            recommendations.append({
                "priority": "HIGH",
                "category": "Performance",
                "issue": f"RTF target not met for {len(failed_tests)}/{len(stage_timing)} test cases",
                "recommendation": "Focus on optimizing the primary bottleneck stage identified in analysis",
                "expected_impact": "50% RTF improvement"
            })

        # Analyze bottlenecks
        bottlenecks = self.audit_results["performance_bottlenecks"]
        primary_bottleneck = bottlenecks["overall_bottleneck"]["stage"]

        if primary_bottleneck == "model_inference":
            recommendations.append({
                "priority": "HIGH",
                "category": "Model Optimization",
                "issue": "Model inference is the primary bottleneck",
                "recommendation": "Implement ONNX Runtime optimization, model quantization, and torch.compile",
                "expected_impact": "30-50% inference speedup"
            })
        elif primary_bottleneck == "preprocessing":
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Preprocessing",
                "issue": "Text preprocessing is taking excessive time",
                "recommendation": "Optimize phonemization caching and text normalization",
                "expected_impact": "20-30% preprocessing speedup"
            })
        elif primary_bottleneck == "audio_processing":
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Audio Processing",
                "issue": "Audio post-processing is the bottleneck",
                "recommendation": "Optimize FFmpeg settings and audio enhancement pipeline",
                "expected_impact": "15-25% processing speedup"
            })

        # Memory analysis recommendations
        memory_analysis = self.audit_results["memory_analysis"]
        potential_leaks = [name for name, data in memory_analysis.items() if data["potential_leak"]]

        if potential_leaks:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Memory Management",
                "issue": f"Potential memory leaks detected in {len(potential_leaks)} test cases",
                "recommendation": "Implement proper memory cleanup and garbage collection optimization",
                "expected_impact": "Improved stability and reduced memory usage"
            })

        # Hardware utilization recommendations
        hw_util = self.audit_results["optimization_effectiveness"]["hardware_utilization"]
        if hw_util["cpu_usage"]["underutilized"]:
            recommendations.append({
                "priority": "LOW",
                "category": "Hardware Utilization",
                "issue": "CPU is underutilized during generation",
                "recommendation": "Implement parallel processing and increase thread pool size",
                "expected_impact": "Better resource utilization"
            })

        self.audit_results["recommendations"] = recommendations

        logger.info(f"Generated {len(recommendations)} optimization recommendations")
        for rec in recommendations:
            logger.info(f"  {rec['priority']}: {rec['issue']}")

    async def _save_audit_results(self):
        """Save audit results to file."""
        output_file = self.output_dir / f"performance_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)

        logger.info(f"Audit results saved to: {output_file}")

        # Generate summary report
        summary_file = self.output_dir / f"audit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        await self._generate_summary_report(summary_file)

    async def _generate_summary_report(self, output_file: Path):
        """Generate a human-readable summary report."""
        stage_timing = self.audit_results["pipeline_analysis"]["stage_timing"]
        bottlenecks = self.audit_results["performance_bottlenecks"]
        recommendations = self.audit_results["recommendations"]

        summary = f"""# JabberTTS Performance Audit Summary
Generated: {self.audit_results['audit_timestamp']}

## Executive Summary
- **Performance Target**: RTF ≤ 0.25
- **Tests Passed**: {sum(1 for data in stage_timing.values() if data['performance_target_met'])}/{len(stage_timing)}
- **Primary Bottleneck**: {bottlenecks['overall_bottleneck']['stage']}
- **Recommendations**: {len(recommendations)} optimization opportunities identified

## Performance Results
"""

        for test_name, data in stage_timing.items():
            status = "✅ PASS" if data["performance_target_met"] else "❌ FAIL"
            summary += f"- **{test_name.title()}**: RTF {data['rtf']:.3f} {status}\n"

        summary += f"""
## Top Bottlenecks
"""
        for stage, avg_time in bottlenecks["average_stage_times"].items():
            summary += f"- **{stage.replace('_', ' ').title()}**: {avg_time:.3f}s average\n"

        summary += f"""
## Priority Recommendations
"""
        for rec in recommendations:
            summary += f"- **{rec['priority']}**: {rec['issue']}\n  - {rec['recommendation']}\n  - Expected impact: {rec['expected_impact']}\n\n"

        with open(output_file, 'w') as f:
            f.write(summary)

        logger.info(f"Summary report saved to: {output_file}")


async def main():
    """Run the comprehensive performance audit."""
    auditor = PerformanceAuditor()
    results = await auditor.run_comprehensive_audit()

    print("\n" + "="*60)
    print("🎯 PERFORMANCE AUDIT COMPLETE")
    print("="*60)

    # Print key findings
    stage_timing = results["pipeline_analysis"]["stage_timing"]
    passed_tests = sum(1 for data in stage_timing.values() if data["performance_target_met"])
    total_tests = len(stage_timing)

    print(f"Performance Target (RTF ≤ 0.25): {passed_tests}/{total_tests} tests passed")
    print(f"Primary Bottleneck: {results['performance_bottlenecks']['overall_bottleneck']['stage']}")
    print(f"Recommendations Generated: {len(results['recommendations'])}")

    return results


if __name__ == "__main__":
    asyncio.run(main())