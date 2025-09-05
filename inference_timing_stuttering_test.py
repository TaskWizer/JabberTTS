#!/usr/bin/env python3
"""Inference Pipeline Timing and Memory Analysis for JabberTTS Stuttering.

This script investigates synchronization issues in async inference, memory allocation
patterns causing audio buffer fragmentation, and model attention mechanisms that
may be causing the T-T-S stuttering artifacts.
"""

import sys
import asyncio
import time
import numpy as np
import json
import gc
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.config import get_settings
from jabbertts.models.manager import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceTimingStutteringTest:
    """Investigate inference pipeline timing and memory allocation issues causing stuttering."""
    
    def __init__(self):
        """Initialize the test."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.model_manager = get_model_manager()
        
        # Create test output directory
        self.output_dir = Path("inference_timing_stuttering_test")
        self.output_dir.mkdir(exist_ok=True)
        
        # Memory and timing tracking
        self.memory_snapshots = []
        self.timing_data = []
        self.process = psutil.Process()
        
        # Test cases for timing analysis
        self.test_cases = [
            {
                "name": "warmup_test",
                "text": "Hello world",
                "description": "Simple warmup test to analyze first inference timing"
            },
            {
                "name": "sequential_test_1",
                "text": "Welcome to the system",
                "description": "First sequential test after warmup"
            },
            {
                "name": "sequential_test_2", 
                "text": "Text-to-speech synthesis",
                "description": "Second sequential test - T-T-S phrase"
            },
            {
                "name": "sequential_test_3",
                "text": "Testing stuttering artifacts",
                "description": "Third sequential test for consistency"
            },
            {
                "name": "concurrent_test",
                "text": "Concurrent processing test",
                "description": "Test concurrent inference behavior"
            }
        ]
        
        print(f"Inference Timing Stuttering Test for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print()
    
    async def run_comprehensive_timing_analysis(self) -> Dict[str, Any]:
        """Run comprehensive timing and memory analysis."""
        print("=== INFERENCE TIMING STUTTERING ANALYSIS ===\n")
        
        results = {
            "test_timestamp": time.time(),
            "model_warmup_analysis": {},
            "sequential_inference_analysis": {},
            "memory_allocation_analysis": {},
            "async_processing_analysis": {},
            "attention_mechanism_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        # Phase 1: Model Warmup Analysis
        print("Phase 1: Model Warmup Analysis")
        print("=" * 50)
        results["model_warmup_analysis"] = await self._analyze_model_warmup()
        
        # Phase 2: Sequential Inference Analysis
        print("\nPhase 2: Sequential Inference Analysis")
        print("=" * 50)
        results["sequential_inference_analysis"] = await self._analyze_sequential_inference()
        
        # Phase 3: Memory Allocation Analysis
        print("\nPhase 3: Memory Allocation Analysis")
        print("=" * 50)
        results["memory_allocation_analysis"] = await self._analyze_memory_allocation()
        
        # Phase 4: Async Processing Analysis
        print("\nPhase 4: Async Processing Analysis")
        print("=" * 50)
        results["async_processing_analysis"] = await self._analyze_async_processing()
        
        # Phase 5: Attention Mechanism Analysis
        print("\nPhase 5: Attention Mechanism Analysis")
        print("=" * 50)
        results["attention_mechanism_analysis"] = await self._analyze_attention_mechanisms()
        
        # Generate findings and recommendations
        results["findings"] = self._generate_timing_findings(results)
        results["recommendations"] = self._generate_timing_recommendations(results)
        
        # Save results
        results_file = self.output_dir / "inference_timing_stuttering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_timing_summary_report(results)
        
        print(f"\n✓ Inference timing stuttering analysis completed!")
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Summary report: {self.output_dir}/INFERENCE_TIMING_STUTTERING_SUMMARY.md")
        
        return results
    
    async def _analyze_model_warmup(self) -> Dict[str, Any]:
        """Analyze model warmup behavior and first inference timing."""
        print("Analyzing model warmup behavior...")
        
        # Ensure clean start
        self.model_manager.unload_model()
        gc.collect()
        
        analysis_results = {
            "cold_start_timing": {},
            "warmup_iterations": [],
            "memory_during_warmup": [],
            "warmup_conclusions": {}
        }
        
        # Cold start analysis
        print("\nCold start analysis...")
        cold_start_data = await self._measure_cold_start()
        analysis_results["cold_start_timing"] = cold_start_data
        
        # Warmup iterations analysis
        print("\nWarmup iterations analysis...")
        for i in range(5):
            print(f"  Warmup iteration {i+1}/5...")
            
            start_time = time.time()
            memory_before = self._get_memory_usage()
            
            result = await self.engine.generate_speech("Warmup test", voice="alloy")
            
            end_time = time.time()
            memory_after = self._get_memory_usage()
            
            iteration_data = {
                "iteration": i + 1,
                "total_time": end_time - start_time,
                "rtf": result["rtf"],
                "duration": result["duration"],
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_after - memory_before,
                "stuttering_score": self._analyze_stuttering_patterns(result["audio_data"])["stuttering_score"]
            }
            
            analysis_results["warmup_iterations"].append(iteration_data)
            
            print(f"    RTF: {result['rtf']:.3f}, Memory: {memory_after:.1f}MB (+{memory_after-memory_before:.1f})")
        
        # Analyze warmup conclusions
        warmup_rtfs = [iter_data["rtf"] for iter_data in analysis_results["warmup_iterations"]]
        warmup_memories = [iter_data["memory_delta"] for iter_data in analysis_results["warmup_iterations"]]
        
        analysis_results["warmup_conclusions"] = {
            "first_iteration_rtf": warmup_rtfs[0],
            "last_iteration_rtf": warmup_rtfs[-1],
            "rtf_improvement": warmup_rtfs[0] - warmup_rtfs[-1],
            "rtf_stabilized": max(warmup_rtfs[-3:]) - min(warmup_rtfs[-3:]) < 0.1,
            "memory_growth_total": sum(warmup_memories),
            "memory_growth_per_iteration": sum(warmup_memories) / len(warmup_memories),
            "memory_leak_detected": any(delta > 50 for delta in warmup_memories)  # >50MB growth
        }
        
        return analysis_results
    
    async def _measure_cold_start(self) -> Dict[str, Any]:
        """Measure detailed cold start timing."""
        cold_start_data = {
            "model_loading_time": 0,
            "first_inference_time": 0,
            "total_cold_start_time": 0,
            "memory_usage": {}
        }
        
        # Measure model loading
        start_time = time.time()
        memory_before_load = self._get_memory_usage()
        
        model = await self.engine._ensure_model_loaded()
        
        load_time = time.time() - start_time
        memory_after_load = self._get_memory_usage()
        
        cold_start_data["model_loading_time"] = load_time
        cold_start_data["memory_usage"]["before_load"] = memory_before_load
        cold_start_data["memory_usage"]["after_load"] = memory_after_load
        cold_start_data["memory_usage"]["load_delta"] = memory_after_load - memory_before_load
        
        # Measure first inference
        start_time = time.time()
        result = await self.engine.generate_speech("Cold start test", voice="alloy")
        inference_time = time.time() - start_time
        memory_after_inference = self._get_memory_usage()
        
        cold_start_data["first_inference_time"] = inference_time
        cold_start_data["first_inference_rtf"] = result["rtf"]
        cold_start_data["total_cold_start_time"] = load_time + inference_time
        cold_start_data["memory_usage"]["after_inference"] = memory_after_inference
        cold_start_data["memory_usage"]["inference_delta"] = memory_after_inference - memory_after_load
        
        return cold_start_data
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def _analyze_stuttering_patterns(self, audio: np.ndarray) -> Dict[str, Any]:
        """Quick stuttering analysis for timing tests."""
        if len(audio) < 1000:
            return {"stuttering_score": 0.0}
        
        # Simple envelope analysis
        window_size = 160  # 10ms at 16kHz
        envelope = np.array([np.max(np.abs(audio[i:i+window_size])) 
                           for i in range(0, len(audio)-window_size, window_size)])
        
        if len(envelope) < 10:
            return {"stuttering_score": 0.0}
        
        # Detect sudden changes and repetitive patterns
        envelope_diff = np.diff(envelope)
        sudden_changes = np.sum(np.abs(envelope_diff) > 0.1)
        
        # Simple stuttering score
        stuttering_score = (sudden_changes / len(envelope)) * 0.5
        
        return {
            "stuttering_score": float(min(stuttering_score, 1.0)),
            "sudden_changes": int(sudden_changes),
            "envelope_length": len(envelope)
        }

    async def _analyze_sequential_inference(self) -> Dict[str, Any]:
        """Analyze sequential inference behavior for timing consistency."""
        print("Analyzing sequential inference behavior...")

        analysis_results = {
            "sequential_tests": [],
            "timing_consistency": {},
            "performance_degradation": {}
        }

        for i, test_case in enumerate(self.test_cases[1:4]):  # Skip warmup, use sequential tests
            print(f"\nSequential test {i+1}: {test_case['name']}")

            # Detailed timing measurement
            timing_breakdown = await self._measure_detailed_inference_timing(test_case["text"])

            # Save audio for analysis
            audio_file = self.output_dir / f"{test_case['name']}_sequential.wav"
            await self._save_audio_sample(
                timing_breakdown["audio_data"],
                timing_breakdown["sample_rate"],
                audio_file
            )

            test_data = {
                "test_name": test_case["name"],
                "text": test_case["text"],
                "audio_file": str(audio_file),
                "timing_breakdown": timing_breakdown,
                "memory_usage": self._get_memory_usage(),
                "stuttering_analysis": self._analyze_stuttering_patterns(timing_breakdown["audio_data"])
            }

            analysis_results["sequential_tests"].append(test_data)

            print(f"    Total RTF: {timing_breakdown['total_rtf']:.3f}")
            print(f"    Preprocessing: {timing_breakdown['preprocessing_time']:.3f}s")
            print(f"    Inference: {timing_breakdown['inference_time']:.3f}s")
            print(f"    Post-processing: {timing_breakdown['postprocessing_time']:.3f}s")
            print(f"    Stuttering score: {test_data['stuttering_analysis']['stuttering_score']:.3f}")

        # Analyze timing consistency
        rtfs = [test["timing_breakdown"]["total_rtf"] for test in analysis_results["sequential_tests"]]
        inference_times = [test["timing_breakdown"]["inference_time"] for test in analysis_results["sequential_tests"]]

        analysis_results["timing_consistency"] = {
            "rtf_variance": float(np.var(rtfs)),
            "rtf_std": float(np.std(rtfs)),
            "rtf_range": float(max(rtfs) - min(rtfs)),
            "inference_time_variance": float(np.var(inference_times)),
            "consistent_performance": np.std(rtfs) < 0.1  # RTF std < 0.1 is consistent
        }

        return analysis_results

    async def _measure_detailed_inference_timing(self, text: str) -> Dict[str, Any]:
        """Measure detailed timing breakdown of inference pipeline."""
        timing_data = {}

        # Total timing
        total_start = time.time()

        # Preprocessing timing
        prep_start = time.time()
        processed_text = await self.engine._preprocess_text(text)
        timing_data["preprocessing_time"] = time.time() - prep_start

        # Model loading timing (should be fast if already loaded)
        load_start = time.time()
        model = await self.engine._ensure_model_loaded()
        timing_data["model_loading_time"] = time.time() - load_start

        # Inference timing with memory tracking
        inf_start = time.time()
        memory_before_inference = self._get_memory_usage()

        audio_data = await self.engine._generate_audio(model, processed_text, "alloy", 1.0)

        timing_data["inference_time"] = time.time() - inf_start
        timing_data["memory_during_inference"] = self._get_memory_usage() - memory_before_inference

        # Post-processing timing
        post_start = time.time()
        audio_bytes, sample_rate = await self.audio_processor.process_audio(
            audio_data, model.get_sample_rate(), "wav"
        )
        timing_data["postprocessing_time"] = time.time() - post_start

        timing_data["total_time"] = time.time() - total_start

        # Calculate RTF
        audio_duration = len(audio_data) / model.get_sample_rate()
        timing_data["audio_duration"] = audio_duration
        timing_data["total_rtf"] = timing_data["total_time"] / audio_duration if audio_duration > 0 else 0
        timing_data["inference_rtf"] = timing_data["inference_time"] / audio_duration if audio_duration > 0 else 0

        # Store audio data for analysis
        timing_data["audio_data"] = audio_data
        timing_data["sample_rate"] = model.get_sample_rate()

        return timing_data

    async def _analyze_memory_allocation(self) -> Dict[str, Any]:
        """Analyze memory allocation patterns during inference."""
        print("Analyzing memory allocation patterns...")

        analysis_results = {
            "memory_snapshots": [],
            "allocation_patterns": {},
            "fragmentation_analysis": {}
        }

        # Start memory monitoring thread
        memory_monitor = MemoryMonitor()
        memory_monitor.start()

        try:
            # Run inference with memory monitoring
            for i in range(3):
                print(f"  Memory test {i+1}/3...")

                memory_before = self._get_memory_usage()
                gc.collect()  # Force garbage collection
                memory_after_gc = self._get_memory_usage()

                result = await self.engine.generate_speech(
                    f"Memory allocation test {i+1}", voice="alloy"
                )

                memory_after = self._get_memory_usage()

                snapshot = {
                    "iteration": i + 1,
                    "memory_before": memory_before,
                    "memory_after_gc": memory_after_gc,
                    "memory_after_inference": memory_after,
                    "gc_freed": memory_before - memory_after_gc,
                    "inference_allocated": memory_after - memory_after_gc,
                    "rtf": result["rtf"],
                    "stuttering_score": self._analyze_stuttering_patterns(result["audio_data"])["stuttering_score"]
                }

                analysis_results["memory_snapshots"].append(snapshot)

                print(f"    Memory: {memory_before:.1f} → {memory_after_gc:.1f} → {memory_after:.1f} MB")
                print(f"    GC freed: {snapshot['gc_freed']:.1f}MB, Inference used: {snapshot['inference_allocated']:.1f}MB")

        finally:
            memory_monitor.stop()
            analysis_results["detailed_memory_trace"] = memory_monitor.get_data()

        # Analyze allocation patterns
        allocations = [snap["inference_allocated"] for snap in analysis_results["memory_snapshots"]]
        gc_freed = [snap["gc_freed"] for snap in analysis_results["memory_snapshots"]]

        analysis_results["allocation_patterns"] = {
            "average_allocation": float(np.mean(allocations)),
            "allocation_variance": float(np.var(allocations)),
            "average_gc_freed": float(np.mean(gc_freed)),
            "memory_leak_detected": any(alloc > 100 for alloc in allocations),  # >100MB per inference
            "gc_effectiveness": float(np.mean(gc_freed)) > 10  # GC frees >10MB on average
        }

        return analysis_results

    async def _analyze_async_processing(self) -> Dict[str, Any]:
        """Analyze async processing behavior and potential synchronization issues."""
        print("Analyzing async processing behavior...")

        analysis_results = {
            "concurrent_tests": [],
            "synchronization_analysis": {},
            "async_performance": {}
        }

        # Test concurrent inference
        print("  Testing concurrent inference...")

        concurrent_tasks = []
        start_time = time.time()

        for i in range(3):
            task = asyncio.create_task(
                self.engine.generate_speech(f"Concurrent test {i+1}", voice="alloy")
            )
            concurrent_tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_concurrent_time = time.time() - start_time

        # Analyze concurrent results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                test_data = {
                    "task_id": i + 1,
                    "error": str(result),
                    "success": False
                }
            else:
                test_data = {
                    "task_id": i + 1,
                    "rtf": result["rtf"],
                    "duration": result["duration"],
                    "stuttering_score": self._analyze_stuttering_patterns(result["audio_data"])["stuttering_score"],
                    "success": True
                }

            analysis_results["concurrent_tests"].append(test_data)

        # Analyze synchronization
        successful_results = [r for r in results if not isinstance(r, Exception)]
        if successful_results:
            rtfs = [r["rtf"] for r in successful_results]

            analysis_results["synchronization_analysis"] = {
                "concurrent_execution_time": total_concurrent_time,
                "average_rtf": float(np.mean(rtfs)),
                "rtf_variance": float(np.var(rtfs)),
                "synchronization_issues": np.var(rtfs) > 0.5,  # High variance indicates sync issues
                "successful_tasks": len(successful_results),
                "failed_tasks": len(results) - len(successful_results)
            }

        return analysis_results

    async def _analyze_attention_mechanisms(self) -> Dict[str, Any]:
        """Analyze model attention mechanisms for potential stuttering causes."""
        print("Analyzing attention mechanisms...")

        analysis_results = {
            "attention_analysis": {},
            "model_behavior": {},
            "attention_patterns": {}
        }

        # Test with different text complexities to analyze attention behavior
        attention_test_cases = [
            {"text": "Hello", "complexity": "simple"},
            {"text": "Text-to-speech", "complexity": "medium"},
            {"text": "Neural network architecture optimization", "complexity": "complex"}
        ]

        for test_case in attention_test_cases:
            print(f"  Analyzing attention for {test_case['complexity']} text...")

            try:
                # Generate with detailed timing
                start_time = time.time()
                result = await self.engine.generate_speech(test_case["text"], voice="alloy")
                generation_time = time.time() - start_time

                # Analyze audio characteristics
                audio_analysis = self._analyze_audio_characteristics(result["audio_data"])

                test_data = {
                    "text": test_case["text"],
                    "complexity": test_case["complexity"],
                    "generation_time": generation_time,
                    "rtf": result["rtf"],
                    "audio_analysis": audio_analysis,
                    "stuttering_score": self._analyze_stuttering_patterns(result["audio_data"])["stuttering_score"]
                }

                analysis_results["attention_patterns"][test_case["complexity"]] = test_data

                print(f"    RTF: {result['rtf']:.3f}, Stuttering: {test_data['stuttering_score']:.3f}")

            except Exception as e:
                print(f"    Error analyzing {test_case['complexity']} text: {e}")
                analysis_results["attention_patterns"][test_case["complexity"]] = {"error": str(e)}

        # Analyze patterns across complexities
        successful_tests = {k: v for k, v in analysis_results["attention_patterns"].items() if "error" not in v}
        if len(successful_tests) >= 2:
            rtfs = [test["rtf"] for test in successful_tests.values()]
            stuttering_scores = [test["stuttering_score"] for test in successful_tests.values()]

            analysis_results["model_behavior"] = {
                "rtf_increases_with_complexity": rtfs[-1] > rtfs[0] * 1.5,
                "stuttering_increases_with_complexity": stuttering_scores[-1] > stuttering_scores[0] * 1.5,
                "attention_degradation": max(stuttering_scores) > 0.2,
                "complexity_correlation": float(np.corrcoef(range(len(rtfs)), rtfs)[0, 1]) if len(rtfs) > 1 else 0
            }

        return analysis_results

    def _analyze_audio_characteristics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio characteristics for attention mechanism insights."""
        if len(audio) < 1000:
            return {"error": "Audio too short for analysis"}

        # Basic audio characteristics
        characteristics = {
            "length": len(audio),
            "duration": len(audio) / 16000,  # Assuming 16kHz
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "peak_amplitude": float(np.max(np.abs(audio))),
            "zero_crossings": int(np.sum(np.diff(np.sign(audio)) != 0)),
            "dynamic_range": float(np.max(audio) - np.min(audio))
        }

        # Spectral characteristics (simplified)
        window_size = 512
        if len(audio) > window_size:
            # Simple spectral analysis
            windowed_audio = audio[:len(audio)//window_size * window_size].reshape(-1, window_size)
            spectral_centroids = []

            for window in windowed_audio:
                fft = np.fft.fft(window)
                magnitude = np.abs(fft[:window_size//2])
                freqs = np.fft.fftfreq(window_size, 1/16000)[:window_size//2]

                if np.sum(magnitude) > 0:
                    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                    spectral_centroids.append(centroid)

            if spectral_centroids:
                characteristics["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
                characteristics["spectral_centroid_std"] = float(np.std(spectral_centroids))
                characteristics["spectral_stability"] = float(np.std(spectral_centroids)) < 500  # Stable if std < 500Hz

        return characteristics

    async def _save_audio_sample(self, audio_data: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Save audio sample for analysis."""
        audio_bytes, _ = await self.audio_processor.process_audio(
            audio_data, sample_rate, "wav"
        )
        with open(filename, 'wb') as f:
            f.write(audio_bytes)

    def _generate_timing_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate findings based on timing analysis."""
        findings = []

        # Model warmup findings
        warmup_analysis = results.get("model_warmup_analysis", {})
        warmup_conclusions = warmup_analysis.get("warmup_conclusions", {})

        if warmup_conclusions.get("first_iteration_rtf", 0) > 5.0:
            findings.append(
                f"CRITICAL: First inference extremely slow - RTF: {warmup_conclusions['first_iteration_rtf']:.3f} "
                f"(improvement: {warmup_conclusions.get('rtf_improvement', 0):.3f})"
            )

        if warmup_conclusions.get("memory_leak_detected", False):
            findings.append(
                f"CRITICAL: Memory leak detected during warmup - "
                f"total growth: {warmup_conclusions.get('memory_growth_total', 0):.1f}MB"
            )

        # Sequential inference findings
        sequential_analysis = results.get("sequential_inference_analysis", {})
        timing_consistency = sequential_analysis.get("timing_consistency", {})

        if not timing_consistency.get("consistent_performance", True):
            findings.append(
                f"WARNING: Inconsistent performance detected - RTF std: {timing_consistency.get('rtf_std', 0):.3f}"
            )

        # Memory allocation findings
        memory_analysis = results.get("memory_allocation_analysis", {})
        allocation_patterns = memory_analysis.get("allocation_patterns", {})

        if allocation_patterns.get("memory_leak_detected", False):
            findings.append(
                f"CRITICAL: Memory leak in inference - average allocation: {allocation_patterns.get('average_allocation', 0):.1f}MB"
            )

        # Async processing findings
        async_analysis = results.get("async_processing_analysis", {})
        sync_analysis = async_analysis.get("synchronization_analysis", {})

        if sync_analysis.get("synchronization_issues", False):
            findings.append(
                f"WARNING: Synchronization issues detected - RTF variance: {sync_analysis.get('rtf_variance', 0):.3f}"
            )

        # Attention mechanism findings
        attention_analysis = results.get("attention_mechanism_analysis", {})
        model_behavior = attention_analysis.get("model_behavior", {})

        if model_behavior.get("attention_degradation", False):
            findings.append(
                "WARNING: Attention mechanism degradation detected with complex text"
            )

        if not findings:
            findings.append("No critical timing or memory issues detected in automated analysis")

        return findings

    def _generate_timing_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on timing analysis."""
        recommendations = []

        # Model warmup recommendations
        warmup_analysis = results.get("model_warmup_analysis", {})
        warmup_conclusions = warmup_analysis.get("warmup_conclusions", {})

        if warmup_conclusions.get("first_iteration_rtf", 0) > 5.0:
            recommendations.append(
                "IMMEDIATE: Implement model warmup during startup to avoid first-inference penalty"
            )
            recommendations.append(
                "OPTIMIZE: Investigate torch.compile warmup and model caching strategies"
            )

        if warmup_conclusions.get("memory_leak_detected", False):
            recommendations.append(
                "CRITICAL: Fix memory leaks in model loading/inference pipeline"
            )

        # Performance recommendations
        sequential_analysis = results.get("sequential_inference_analysis", {})
        timing_consistency = sequential_analysis.get("timing_consistency", {})

        if not timing_consistency.get("consistent_performance", True):
            recommendations.append(
                "INVESTIGATE: Implement consistent memory management and garbage collection"
            )

        # Memory recommendations
        memory_analysis = results.get("memory_allocation_analysis", {})
        allocation_patterns = memory_analysis.get("allocation_patterns", {})

        if allocation_patterns.get("average_allocation", 0) > 50:
            recommendations.append(
                "OPTIMIZE: Reduce memory allocation per inference through tensor reuse"
            )

        # General recommendations
        recommendations.extend([
            "IMPLEMENT: Model warmup routine during application startup",
            "MONITOR: Add real-time memory and timing monitoring to production",
            "OPTIMIZE: Implement tensor caching and memory pool management",
            "VALIDATE: Test with production-scale concurrent load"
        ])

        return recommendations

    def _generate_timing_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive timing analysis summary report."""
        report_file = self.output_dir / "INFERENCE_TIMING_STUTTERING_SUMMARY.md"

        with open(report_file, 'w') as f:
            f.write("# Inference Timing Stuttering Analysis Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analysis Type**: Inference Pipeline Timing and Memory Analysis\n")
            f.write(f"**Model**: {self.settings.model_name}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This analysis investigated synchronization issues in async inference, ")
            f.write("memory allocation patterns causing audio buffer fragmentation, and model ")
            f.write("attention mechanisms that may be causing T-T-S stuttering artifacts.\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            findings = results.get("findings", [])
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            recommendations = results.get("recommendations", [])
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Model Warmup Analysis
            warmup_analysis = results.get("model_warmup_analysis", {})
            if "warmup_conclusions" in warmup_analysis:
                f.write("## Model Warmup Analysis\n\n")
                conclusions = warmup_analysis["warmup_conclusions"]
                f.write(f"- **First Iteration RTF**: {conclusions.get('first_iteration_rtf', 0):.3f}\n")
                f.write(f"- **Last Iteration RTF**: {conclusions.get('last_iteration_rtf', 0):.3f}\n")
                f.write(f"- **RTF Improvement**: {conclusions.get('rtf_improvement', 0):.3f}\n")
                f.write(f"- **Performance Stabilized**: {conclusions.get('rtf_stabilized', False)}\n")
                f.write(f"- **Memory Growth Total**: {conclusions.get('memory_growth_total', 0):.1f}MB\n")
                f.write(f"- **Memory Leak Detected**: {conclusions.get('memory_leak_detected', False)}\n\n")

            # Sequential Inference Analysis
            sequential_analysis = results.get("sequential_inference_analysis", {})
            if "timing_consistency" in sequential_analysis:
                f.write("## Sequential Inference Analysis\n\n")
                consistency = sequential_analysis["timing_consistency"]
                f.write(f"- **RTF Variance**: {consistency.get('rtf_variance', 0):.6f}\n")
                f.write(f"- **RTF Standard Deviation**: {consistency.get('rtf_std', 0):.3f}\n")
                f.write(f"- **RTF Range**: {consistency.get('rtf_range', 0):.3f}\n")
                f.write(f"- **Consistent Performance**: {consistency.get('consistent_performance', False)}\n\n")

            # Memory Allocation Analysis
            memory_analysis = results.get("memory_allocation_analysis", {})
            if "allocation_patterns" in memory_analysis:
                f.write("## Memory Allocation Analysis\n\n")
                patterns = memory_analysis["allocation_patterns"]
                f.write(f"- **Average Allocation**: {patterns.get('average_allocation', 0):.1f}MB\n")
                f.write(f"- **Allocation Variance**: {patterns.get('allocation_variance', 0):.3f}\n")
                f.write(f"- **Average GC Freed**: {patterns.get('average_gc_freed', 0):.1f}MB\n")
                f.write(f"- **Memory Leak Detected**: {patterns.get('memory_leak_detected', False)}\n")
                f.write(f"- **GC Effectiveness**: {patterns.get('gc_effectiveness', False)}\n\n")

            # Test Results Table
            if "sequential_tests" in sequential_analysis:
                f.write("## Sequential Test Results\n\n")
                f.write("| Test | RTF | Preprocessing | Inference | Post-processing | Stuttering Score |\n")
                f.write("|------|-----|---------------|-----------|-----------------|------------------|\n")

                for test in sequential_analysis["sequential_tests"]:
                    timing = test["timing_breakdown"]
                    f.write(f"| {test['test_name']} | {timing['total_rtf']:.3f} | ")
                    f.write(f"{timing['preprocessing_time']:.3f}s | ")
                    f.write(f"{timing['inference_time']:.3f}s | ")
                    f.write(f"{timing['postprocessing_time']:.3f}s | ")
                    f.write(f"{test['stuttering_analysis']['stuttering_score']:.3f} |\n")
                f.write("\n")

            # Generated Audio Files
            f.write("## Generated Audio Files\n\n")
            f.write("Analyze these audio files for timing-related stuttering artifacts:\n\n")

            if "sequential_tests" in sequential_analysis:
                for test in sequential_analysis["sequential_tests"]:
                    f.write(f"- **{test['test_name']}**: `{test['test_name']}_sequential.wav`\n")
                    f.write(f"  - Text: \"{test['text']}\"\n")
                    f.write(f"  - RTF: {test['timing_breakdown']['total_rtf']:.3f}\n")
                    f.write(f"  - Stuttering Score: {test['stuttering_analysis']['stuttering_score']:.3f}\n\n")

            f.write("---\n")
            f.write("**Note**: Manual audio analysis is essential to confirm timing-related stuttering causes.\n")


class MemoryMonitor:
    """Thread-based memory monitoring for detailed allocation tracking."""

    def __init__(self, interval: float = 0.1):
        """Initialize memory monitor with sampling interval."""
        self.interval = interval
        self.running = False
        self.thread = None
        self.data = []
        self.process = psutil.Process()

    def start(self):
        """Start memory monitoring."""
        self.running = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        """Stop memory monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

    def get_data(self) -> List[Dict[str, Any]]:
        """Get collected memory data."""
        return self.data.copy()

    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.running:
            try:
                memory_info = self.process.memory_info()
                self.data.append({
                    "timestamp": time.time(),
                    "rss": memory_info.rss / 1024 / 1024,  # MB
                    "vms": memory_info.vms / 1024 / 1024,  # MB
                })
            except Exception as e:
                # Ignore monitoring errors
                pass

            time.sleep(self.interval)


async def main():
    """Execute the complete inference timing analysis workflow."""
    test = InferenceTimingStutteringTest()

    try:
        print("🔍 Starting comprehensive inference timing stuttering analysis...")
        print("This test will investigate timing, memory, and synchronization issues")
        print("that may be causing T-T-S stuttering artifacts.\n")

        results = await test.run_comprehensive_timing_analysis()

        print("\n" + "="*70)
        print("INFERENCE TIMING STUTTERING ANALYSIS COMPLETED")
        print("="*70)

        # Print key findings
        findings = results.get("findings", [])
        if findings:
            print("\n🔍 KEY FINDINGS:")
            for finding in findings:
                print(f"  • {finding}")

        # Print top recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")

        # Print warmup analysis summary
        warmup_analysis = results.get("model_warmup_analysis", {})
        warmup_conclusions = warmup_analysis.get("warmup_conclusions", {})
        if warmup_conclusions:
            print(f"\n📊 WARMUP ANALYSIS:")
            print(f"  • First inference RTF: {warmup_conclusions.get('first_iteration_rtf', 0):.3f}")
            print(f"  • RTF improvement: {warmup_conclusions.get('rtf_improvement', 0):.3f}")
            print(f"  • Memory growth: {warmup_conclusions.get('memory_growth_total', 0):.1f}MB")
            print(f"  • Performance stabilized: {warmup_conclusions.get('rtf_stabilized', False)}")

        print(f"\n📁 Test files saved to: {test.output_dir}")
        print("📋 Review the summary report and listen to audio samples")
        print("\n🎧 MANUAL LISTENING TEST REQUIRED:")
        print("   Listen to sequential test audio files to identify timing-related stuttering")

        return results

    except Exception as e:
        print(f"\n❌ Inference timing stuttering analysis failed: {e}")
        logger.exception("Analysis failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
