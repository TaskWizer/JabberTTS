#!/usr/bin/env python3
"""Torch Compile Stuttering Test for JabberTTS.

This script specifically tests whether torch.compile optimization is causing
stuttering artifacts by comparing compiled vs non-compiled model outputs.
"""

import sys
import asyncio
import time
import numpy as np
import json
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


class TorchCompileStutteringTest:
    """Test torch.compile impact on stuttering artifacts."""
    
    def __init__(self):
        """Initialize the test."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.model_manager = get_model_manager()
        
        # Create test output directory
        self.output_dir = Path("torch_compile_stuttering_test")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test cases specifically for stuttering detection
        self.test_cases = [
            {
                "name": "welcome_word",
                "text": "Welcome",
                "description": "Single word reported as intelligible"
            },
            {
                "name": "tts_phrase", 
                "text": "Text-to-speech synthesis",
                "description": "Phrase that triggers T-T-S stuttering"
            },
            {
                "name": "stuttering_test",
                "text": "Testing text-to-speech stuttering artifacts",
                "description": "Phrase designed to trigger stuttering"
            }
        ]
        
        print(f"Torch Compile Stuttering Test for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print()
    
    async def run_compile_comparison_test(self) -> Dict[str, Any]:
        """Run comparison test between compiled and non-compiled models."""
        print("=== TORCH COMPILE STUTTERING TEST ===\n")
        
        results = {
            "test_timestamp": time.time(),
            "compiled_model_results": {},
            "non_compiled_model_results": {},
            "comparison_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        # Phase 1: Test with torch.compile ENABLED (current state)
        print("Phase 1: Testing with torch.compile ENABLED")
        print("=" * 50)
        results["compiled_model_results"] = await self._test_with_compile_enabled()
        
        # Phase 2: Test with torch.compile DISABLED
        print("\nPhase 2: Testing with torch.compile DISABLED")
        print("=" * 50)
        results["non_compiled_model_results"] = await self._test_with_compile_disabled()
        
        # Phase 3: Compare results
        print("\nPhase 3: Analyzing Differences")
        print("=" * 50)
        results["comparison_analysis"] = self._compare_compile_results(
            results["compiled_model_results"],
            results["non_compiled_model_results"]
        )
        
        # Generate findings and recommendations
        results["findings"] = self._generate_compile_findings(results)
        results["recommendations"] = self._generate_compile_recommendations(results)
        
        # Save results
        results_file = self.output_dir / "torch_compile_stuttering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_compile_summary_report(results)
        
        print(f"\n✓ Torch compile stuttering test completed!")
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Summary report: {self.output_dir}/TORCH_COMPILE_STUTTERING_SUMMARY.md")
        
        return results
    
    async def _test_with_compile_enabled(self) -> Dict[str, Any]:
        """Test with torch.compile enabled (current state)."""
        print("Testing with torch.compile ENABLED...")
        
        # Ensure model is loaded and compiled
        model = await self.engine._ensure_model_loaded()
        is_compiled = hasattr(model.model, '_orig_mod')
        
        print(f"Model compilation status: {is_compiled}")
        
        results = {
            "compilation_status": is_compiled,
            "test_results": {}
        }
        
        for test_case in self.test_cases:
            print(f"\nTesting: {test_case['name']} - '{test_case['text']}'")
            
            try:
                # Generate speech with compiled model
                start_time = time.time()
                result = await self.engine.generate_speech(test_case["text"], voice="alloy")
                generation_time = time.time() - start_time
                
                # Save audio sample
                audio_file = self.output_dir / f"{test_case['name']}_compiled.wav"
                await self._save_audio_sample(
                    result["audio_data"], 
                    result["sample_rate"],
                    audio_file
                )
                
                # Analyze audio for stuttering patterns
                stuttering_analysis = self._analyze_stuttering_patterns(result["audio_data"])
                
                results["test_results"][test_case["name"]] = {
                    "audio_file": str(audio_file),
                    "duration": result["duration"],
                    "rtf": result["rtf"],
                    "generation_time": generation_time,
                    "sample_rate": result["sample_rate"],
                    "stuttering_analysis": stuttering_analysis
                }
                
                print(f"    ✓ Compiled: RTF={result['rtf']:.3f}, Duration={result['duration']:.2f}s")
                print(f"    ✓ Stuttering score: {stuttering_analysis['stuttering_score']:.3f}")
                
            except Exception as e:
                print(f"    ✗ Compiled test failed: {e}")
                results["test_results"][test_case["name"]] = {"error": str(e)}
        
        return results
    
    async def _test_with_compile_disabled(self) -> Dict[str, Any]:
        """Test with torch.compile disabled."""
        print("Testing with torch.compile DISABLED...")
        
        # Reload model without compilation
        print("Reloading model without torch.compile...")
        
        # Unload current model
        self.model_manager.unload_model()
        
        # Temporarily disable compilation
        import torch
        original_compile = torch.compile
        torch.compile = lambda model, *args, **kwargs: model  # No-op compile
        
        try:
            # Load model without compilation
            model = await self.engine._ensure_model_loaded()
            is_compiled = hasattr(model.model, '_orig_mod')
            
            print(f"Model compilation status: {is_compiled}")
            
            results = {
                "compilation_status": is_compiled,
                "test_results": {}
            }
            
            for test_case in self.test_cases:
                print(f"\nTesting: {test_case['name']} - '{test_case['text']}'")
                
                try:
                    # Generate speech with non-compiled model
                    start_time = time.time()
                    result = await self.engine.generate_speech(test_case["text"], voice="alloy")
                    generation_time = time.time() - start_time
                    
                    # Save audio sample
                    audio_file = self.output_dir / f"{test_case['name']}_non_compiled.wav"
                    await self._save_audio_sample(
                        result["audio_data"], 
                        result["sample_rate"],
                        audio_file
                    )
                    
                    # Analyze audio for stuttering patterns
                    stuttering_analysis = self._analyze_stuttering_patterns(result["audio_data"])
                    
                    results["test_results"][test_case["name"]] = {
                        "audio_file": str(audio_file),
                        "duration": result["duration"],
                        "rtf": result["rtf"],
                        "generation_time": generation_time,
                        "sample_rate": result["sample_rate"],
                        "stuttering_analysis": stuttering_analysis
                    }
                    
                    print(f"    ✓ Non-compiled: RTF={result['rtf']:.3f}, Duration={result['duration']:.2f}s")
                    print(f"    ✓ Stuttering score: {stuttering_analysis['stuttering_score']:.3f}")
                    
                except Exception as e:
                    print(f"    ✗ Non-compiled test failed: {e}")
                    results["test_results"][test_case["name"]] = {"error": str(e)}
            
        finally:
            # Restore original torch.compile
            torch.compile = original_compile
        
        return results
    
    def _analyze_stuttering_patterns(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio for stuttering patterns."""
        analysis = {
            "stuttering_score": 0.0,
            "repetitive_patterns": 0,
            "sudden_amplitude_changes": 0,
            "silence_gaps": 0,
            "potential_stuttering": False
        }
        
        if len(audio) < 1000:
            return analysis
        
        # Analyze amplitude envelope for stuttering patterns
        window_size = 160  # 10ms at 16kHz
        envelope = np.array([np.max(np.abs(audio[i:i+window_size])) 
                           for i in range(0, len(audio)-window_size, window_size)])
        
        if len(envelope) < 10:
            return analysis
        
        # Detect sudden amplitude changes (potential stuttering)
        envelope_diff = np.diff(envelope)
        sudden_changes = np.sum(np.abs(envelope_diff) > 0.1)
        analysis["sudden_amplitude_changes"] = int(sudden_changes)
        
        # Detect repetitive patterns
        # Look for similar amplitude patterns that repeat
        pattern_length = 5  # 50ms patterns
        repetitive_count = 0
        
        for i in range(len(envelope) - pattern_length * 2):
            pattern1 = envelope[i:i+pattern_length]
            pattern2 = envelope[i+pattern_length:i+pattern_length*2]
            
            # Calculate correlation
            if np.std(pattern1) > 0.01 and np.std(pattern2) > 0.01:
                correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                if correlation > 0.8:  # High correlation indicates repetition
                    repetitive_count += 1
        
        analysis["repetitive_patterns"] = repetitive_count
        
        # Detect silence gaps (potential stuttering artifacts)
        silence_threshold = 0.01
        silence_mask = envelope < silence_threshold
        silence_gaps = 0
        
        in_silence = False
        for is_silent in silence_mask:
            if is_silent and not in_silence:
                silence_gaps += 1
                in_silence = True
            elif not is_silent:
                in_silence = False
        
        analysis["silence_gaps"] = silence_gaps
        
        # Calculate overall stuttering score
        stuttering_score = (
            (sudden_changes / len(envelope)) * 0.4 +
            (repetitive_count / max(len(envelope) - pattern_length * 2, 1)) * 0.4 +
            (silence_gaps / max(len(envelope) // 10, 1)) * 0.2
        )
        
        analysis["stuttering_score"] = float(stuttering_score)
        analysis["potential_stuttering"] = stuttering_score > 0.1
        
        return analysis
    
    async def _save_audio_sample(self, audio_data: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Save audio sample as WAV."""
        audio_bytes, _ = await self.audio_processor.process_audio(
            audio_data, sample_rate, "wav"
        )
        with open(filename, 'wb') as f:
            f.write(audio_bytes)

    def _compare_compile_results(self, compiled_results: Dict, non_compiled_results: Dict) -> Dict[str, Any]:
        """Compare results between compiled and non-compiled models."""
        comparison = {
            "performance_differences": {},
            "stuttering_differences": {},
            "compilation_impact": {}
        }

        for test_name in compiled_results.get("test_results", {}):
            if test_name in non_compiled_results.get("test_results", {}):
                compiled = compiled_results["test_results"][test_name]
                non_compiled = non_compiled_results["test_results"][test_name]

                if "error" not in compiled and "error" not in non_compiled:
                    # Performance comparison
                    rtf_diff = compiled["rtf"] - non_compiled["rtf"]
                    duration_diff = compiled["duration"] - non_compiled["duration"]

                    comparison["performance_differences"][test_name] = {
                        "compiled_rtf": compiled["rtf"],
                        "non_compiled_rtf": non_compiled["rtf"],
                        "rtf_difference": rtf_diff,
                        "rtf_improvement": rtf_diff < 0,
                        "compiled_duration": compiled["duration"],
                        "non_compiled_duration": non_compiled["duration"],
                        "duration_difference": duration_diff
                    }

                    # Stuttering comparison
                    compiled_stuttering = compiled["stuttering_analysis"]["stuttering_score"]
                    non_compiled_stuttering = non_compiled["stuttering_analysis"]["stuttering_score"]
                    stuttering_diff = compiled_stuttering - non_compiled_stuttering

                    comparison["stuttering_differences"][test_name] = {
                        "compiled_stuttering_score": compiled_stuttering,
                        "non_compiled_stuttering_score": non_compiled_stuttering,
                        "stuttering_difference": stuttering_diff,
                        "compilation_increases_stuttering": stuttering_diff > 0.05,
                        "compiled_potential_stuttering": compiled["stuttering_analysis"]["potential_stuttering"],
                        "non_compiled_potential_stuttering": non_compiled["stuttering_analysis"]["potential_stuttering"]
                    }

        # Overall impact assessment
        rtf_diffs = [comp["rtf_difference"] for comp in comparison["performance_differences"].values()]
        stuttering_diffs = [comp["stuttering_difference"] for comp in comparison["stuttering_differences"].values()]

        if rtf_diffs and stuttering_diffs:
            comparison["compilation_impact"] = {
                "average_rtf_change": sum(rtf_diffs) / len(rtf_diffs),
                "average_stuttering_change": sum(stuttering_diffs) / len(stuttering_diffs),
                "compilation_improves_performance": sum(rtf_diffs) < 0,
                "compilation_increases_stuttering": sum(stuttering_diffs) > 0,
                "significant_stuttering_impact": any(abs(diff) > 0.05 for diff in stuttering_diffs)
            }

        return comparison

    def _generate_compile_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate findings based on torch.compile comparison."""
        findings = []

        comparison = results.get("comparison_analysis", {})
        impact = comparison.get("compilation_impact", {})

        if impact.get("compilation_increases_stuttering", False):
            findings.append(
                f"CRITICAL: torch.compile increases stuttering artifacts by average of "
                f"{impact.get('average_stuttering_change', 0):.3f}"
            )

        if impact.get("significant_stuttering_impact", False):
            findings.append(
                "CRITICAL: torch.compile has significant impact on stuttering (>0.05 difference)"
            )

        # Check individual test cases
        for test_name, stuttering_data in comparison.get("stuttering_differences", {}).items():
            if stuttering_data.get("compilation_increases_stuttering", False):
                findings.append(
                    f"WARNING: torch.compile increases stuttering for '{test_name}' - "
                    f"score: {stuttering_data['compiled_stuttering_score']:.3f} vs "
                    f"{stuttering_data['non_compiled_stuttering_score']:.3f}"
                )

        if not findings:
            findings.append("torch.compile does not appear to be causing stuttering artifacts")

        return findings

    def _generate_compile_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on torch.compile analysis."""
        recommendations = []

        comparison = results.get("comparison_analysis", {})
        impact = comparison.get("compilation_impact", {})

        if impact.get("compilation_increases_stuttering", False):
            recommendations.append(
                "IMMEDIATE: Disable torch.compile in production to eliminate stuttering"
            )
            recommendations.append(
                "INVESTIGATE: Review torch.compile configuration for speech generation compatibility"
            )
        else:
            recommendations.append(
                "CONTINUE: torch.compile can be safely used - stuttering cause is elsewhere"
            )

        if impact.get("compilation_improves_performance", True):
            recommendations.append(
                "OPTIMIZE: torch.compile provides performance benefits - keep enabled if no stuttering"
            )

        recommendations.extend([
            "VALIDATE: Generate human listening test samples with torch.compile disabled",
            "MONITOR: Implement stuttering detection in production pipeline",
            "DOCUMENT: Update model configuration guidelines based on findings"
        ])

        return recommendations

    def _generate_compile_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate summary report for torch.compile analysis."""
        report_file = self.output_dir / "TORCH_COMPILE_STUTTERING_SUMMARY.md"

        with open(report_file, 'w') as f:
            f.write("# Torch Compile Stuttering Analysis Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analysis Type**: torch.compile Impact on Stuttering Artifacts\n")
            f.write(f"**Model**: {self.settings.model_name}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This analysis specifically tested whether PyTorch's torch.compile optimization ")
            f.write("is causing stuttering artifacts in JabberTTS speech generation.\n\n")

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

            # Performance Impact
            comparison = results.get("comparison_analysis", {})
            impact = comparison.get("compilation_impact", {})
            if impact:
                f.write("## torch.compile Impact Analysis\n\n")
                f.write(f"- **Average RTF Change**: {impact.get('average_rtf_change', 0):.3f}\n")
                f.write(f"- **Average Stuttering Change**: {impact.get('average_stuttering_change', 0):.3f}\n")
                f.write(f"- **Performance Improvement**: {impact.get('compilation_improves_performance', False)}\n")
                f.write(f"- **Increases Stuttering**: {impact.get('compilation_increases_stuttering', False)}\n\n")

            # Test Results Summary
            f.write("## Test Results Summary\n\n")
            f.write("| Test Case | Compiled RTF | Non-Compiled RTF | Compiled Stuttering | Non-Compiled Stuttering |\n")
            f.write("|-----------|--------------|------------------|---------------------|-------------------------|\n")

            for test_name, perf_data in comparison.get("performance_differences", {}).items():
                stuttering_data = comparison.get("stuttering_differences", {}).get(test_name, {})
                f.write(f"| {test_name} | {perf_data.get('compiled_rtf', 0):.3f} | ")
                f.write(f"{perf_data.get('non_compiled_rtf', 0):.3f} | ")
                f.write(f"{stuttering_data.get('compiled_stuttering_score', 0):.3f} | ")
                f.write(f"{stuttering_data.get('non_compiled_stuttering_score', 0):.3f} |\n")
            f.write("\n")

            f.write("## Generated Audio Files\n\n")
            f.write("Compare these audio files to identify stuttering differences:\n\n")

            for test_case in self.test_cases:
                f.write(f"### {test_case['name'].replace('_', ' ').title()}\n")
                f.write(f"**Text**: \"{test_case['text']}\"\n")
                f.write(f"- Compiled: `{test_case['name']}_compiled.wav`\n")
                f.write(f"- Non-compiled: `{test_case['name']}_non_compiled.wav`\n\n")

            f.write("---\n")
            f.write("**Note**: Manual audio comparison is essential to confirm automated analysis.\n")


async def main():
    """Run the torch compile stuttering test."""
    test = TorchCompileStutteringTest()

    try:
        results = await test.run_compile_comparison_test()

        print("\n" + "="*60)
        print("TORCH COMPILE STUTTERING TEST COMPLETED")
        print("="*60)

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

        print(f"\n📁 Test files saved to: {test.output_dir}")
        print("📋 Review the summary report and listen to audio samples")

        return results

    except Exception as e:
        print(f"\n❌ Torch compile stuttering test failed: {e}")
        logger.exception("Test failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
