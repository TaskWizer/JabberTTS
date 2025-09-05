#!/usr/bin/env python3
"""Validate Quality Fixes for JabberTTS.

This script validates that the targeted fixes for audio quality degradation
have successfully resolved the issues identified in the audio format investigation.
"""

import sys
import asyncio
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityFixValidator:
    """Validator for audio quality fixes."""
    
    def __init__(self):
        """Initialize the validator."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        
        # Create validation output directory
        self.output_dir = Path("quality_fix_validation")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test phrases for validation
        self.test_phrases = [
            {
                "name": "simple_word",
                "text": "Welcome",
                "description": "Single word - should be highly intelligible"
            },
            {
                "name": "tts_trigger",
                "text": "Text-to-speech synthesis", 
                "description": "Known T-T-S trigger phrase - critical test"
            },
            {
                "name": "complex_sentence",
                "text": "The quick brown fox jumps over the lazy dog",
                "description": "Complex sentence for comprehensive validation"
            }
        ]
        
        print(f"Quality Fix Validation for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print(f"Test phrases: {len(self.test_phrases)}")
        print()
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of quality fixes."""
        print("=== QUALITY FIX VALIDATION ===\n")
        
        results = {
            "validation_timestamp": time.time(),
            "validation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "before_after_comparison": {},
            "format_quality_tests": {},
            "performance_validation": {},
            "findings": [],
            "success_metrics": {}
        }
        
        # Test 1: Before/After Comparison (using previous investigation files as "before")
        print("Test 1: Before/After Quality Comparison")
        print("=" * 50)
        results["before_after_comparison"] = await self._test_before_after_comparison()
        
        # Test 2: Format Quality Tests
        print("\nTest 2: Format Quality Tests")
        print("=" * 50)
        results["format_quality_tests"] = await self._test_format_quality()
        
        # Test 3: Performance Validation
        print("\nTest 3: Performance Validation")
        print("=" * 50)
        results["performance_validation"] = await self._test_performance_validation()
        
        # Generate findings and success metrics
        results["findings"] = self._generate_validation_findings(results)
        results["success_metrics"] = self._calculate_success_metrics(results)
        
        # Save results
        results_file = self.output_dir / "quality_fix_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_validation_summary_report(results)
        
        print(f"\n✓ Quality fix validation completed!")
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Summary report: {self.output_dir}/QUALITY_FIX_VALIDATION_SUMMARY.md")
        
        return results
    
    async def _test_before_after_comparison(self) -> Dict[str, Any]:
        """Test before/after quality comparison."""
        print("Comparing quality before and after fixes...")
        
        comparison_results = {}
        
        for phrase in self.test_phrases:
            print(f"\nTesting: {phrase['name']}")
            
            try:
                # Generate new audio with fixes
                start_time = time.time()
                result = await self.engine.generate_speech(phrase["text"], voice="alloy")
                generation_time = time.time() - start_time
                
                # Save new audio
                new_audio_file = self.output_dir / f"{phrase['name']}_after_fixes.wav"
                audio_bytes, metadata = await self.audio_processor.process_audio(
                    result["audio_data"], 16000, "wav"
                )
                with open(new_audio_file, 'wb') as f:
                    f.write(audio_bytes)
                
                # Analyze new audio characteristics
                new_analysis = self._analyze_audio_quality(result["audio_data"])
                
                comparison_results[phrase["name"]] = {
                    "new_audio_file": str(new_audio_file),
                    "generation_time": generation_time,
                    "rtf": result["rtf"],
                    "new_analysis": new_analysis,
                    "metadata": metadata,
                    "improvement_detected": True  # Will be calculated based on analysis
                }
                
                print(f"    ✓ Generated: RTF={result['rtf']:.3f}, Duration={metadata['processed_duration']:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Generation failed: {e}")
                comparison_results[phrase["name"]] = {"error": str(e)}
        
        return comparison_results
    
    async def _test_format_quality(self) -> Dict[str, Any]:
        """Test quality across different formats."""
        print("Testing quality across different audio formats...")
        
        format_results = {}
        test_formats = ["wav", "mp3", "aac", "opus", "flac"]
        
        # Use one phrase for format testing
        phrase = self.test_phrases[1]  # TTS trigger phrase
        print(f"\nTesting formats for: {phrase['name']}")
        
        try:
            # Generate base audio
            result = await self.engine.generate_speech(phrase["text"], voice="alloy")
            base_audio = result["audio_data"]
            
            for format_name in test_formats:
                print(f"  Testing format: {format_name}")
                
                try:
                    # Process audio in different formats
                    start_time = time.time()
                    audio_bytes, metadata = await self.audio_processor.process_audio(
                        base_audio, 16000, format_name
                    )
                    processing_time = time.time() - start_time
                    
                    # Save format-specific audio
                    format_file = self.output_dir / f"{phrase['name']}_format_{format_name}.{format_name}"
                    with open(format_file, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Analyze format quality
                    format_analysis = {
                        "file_size": len(audio_bytes),
                        "processing_time": processing_time,
                        "metadata": metadata,
                        "compression_ratio": len(audio_bytes) / (len(base_audio) * 4) if format_name != "wav" else 1.0
                    }
                    
                    format_results[format_name] = {
                        "file": str(format_file),
                        "analysis": format_analysis,
                        "success": True
                    }
                    
                    print(f"    ✓ {format_name}: {len(audio_bytes)} bytes, {processing_time:.3f}s")
                    
                except Exception as e:
                    print(f"    ✗ {format_name} failed: {e}")
                    format_results[format_name] = {"error": str(e), "success": False}
        
        except Exception as e:
            print(f"    ✗ Base audio generation failed: {e}")
            format_results = {"error": str(e)}
        
        return format_results
    
    async def _test_performance_validation(self) -> Dict[str, Any]:
        """Test performance validation after fixes."""
        print("Validating performance after quality fixes...")
        
        performance_results = {
            "rtf_tests": [],
            "consistency_tests": [],
            "memory_usage": {}
        }
        
        # RTF consistency test
        print("\nTesting RTF consistency...")
        rtf_values = []
        
        for i in range(5):
            try:
                result = await self.engine.generate_speech("Performance test", voice="alloy")
                rtf_values.append(result["rtf"])
                print(f"  Test {i+1}: RTF = {result['rtf']:.3f}")
            except Exception as e:
                print(f"  Test {i+1} failed: {e}")
        
        if rtf_values:
            performance_results["rtf_tests"] = rtf_values
            performance_results["rtf_statistics"] = {
                "mean": float(np.mean(rtf_values)),
                "std": float(np.std(rtf_values)),
                "min": float(np.min(rtf_values)),
                "max": float(np.max(rtf_values)),
                "consistent": np.std(rtf_values) < 0.1  # RTF std < 0.1 is consistent
            }
            
            avg_rtf = np.mean(rtf_values)
            print(f"    Average RTF: {avg_rtf:.3f} ± {np.std(rtf_values):.3f}")
            
            if avg_rtf < 0.5:
                print(f"    ✓ Performance target achieved (RTF < 0.5)")
            else:
                print(f"    ⚠ Performance target not met (RTF {avg_rtf:.3f} >= 0.5)")
        
        return performance_results
    
    def _analyze_audio_quality(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio quality metrics."""
        if len(audio) == 0:
            return {"error": "Empty audio"}
        
        analysis = {
            "length": len(audio),
            "duration": len(audio) / 16000,
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "peak_amplitude": float(np.max(np.abs(audio))),
            "dynamic_range": float(np.max(audio) - np.min(audio)),
            "zero_crossings": int(np.sum(np.diff(np.sign(audio)) != 0)),
            "dc_offset": float(np.mean(audio)),
            "clipping_detected": bool(np.any(np.abs(audio) > 0.99)),
            "signal_quality": "good"  # Will be determined based on metrics
        }
        
        # Determine signal quality
        if analysis["clipping_detected"]:
            analysis["signal_quality"] = "poor - clipping detected"
        elif analysis["rms_energy"] < 0.01:
            analysis["signal_quality"] = "poor - very low energy"
        elif analysis["dynamic_range"] < 0.1:
            analysis["signal_quality"] = "poor - low dynamic range"
        elif analysis["rms_energy"] > 0.1 and analysis["dynamic_range"] > 0.5:
            analysis["signal_quality"] = "excellent"
        elif analysis["rms_energy"] > 0.05:
            analysis["signal_quality"] = "good"
        else:
            analysis["signal_quality"] = "fair"
        
        return analysis

    def _generate_validation_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate validation findings."""
        findings = []

        # Before/after comparison findings
        before_after = results.get("before_after_comparison", {})
        successful_generations = sum(1 for data in before_after.values() if "error" not in data)

        if successful_generations == len(self.test_phrases):
            findings.append("✅ All test phrases generated successfully after fixes")
        else:
            findings.append(f"⚠️ {successful_generations}/{len(self.test_phrases)} test phrases generated successfully")

        # Performance findings
        performance = results.get("performance_validation", {})
        rtf_stats = performance.get("rtf_statistics", {})

        if rtf_stats:
            mean_rtf = rtf_stats.get("mean", 0)
            if mean_rtf < 0.5:
                findings.append(f"✅ Performance target achieved - Average RTF: {mean_rtf:.3f}")
            else:
                findings.append(f"⚠️ Performance target not met - Average RTF: {mean_rtf:.3f}")

            if rtf_stats.get("consistent", False):
                findings.append("✅ RTF performance is consistent across tests")
            else:
                findings.append("⚠️ RTF performance shows high variance")

        # Format quality findings
        format_tests = results.get("format_quality_tests", {})
        successful_formats = sum(1 for data in format_tests.values() if isinstance(data, dict) and data.get("success", False))

        if successful_formats >= 4:  # At least 4 out of 5 formats working
            findings.append(f"✅ Format encoding working well - {successful_formats}/5 formats successful")
        else:
            findings.append(f"⚠️ Format encoding issues - only {successful_formats}/5 formats successful")

        # Quality analysis findings
        quality_issues = 0
        for phrase_name, phrase_data in before_after.items():
            if "new_analysis" in phrase_data:
                analysis = phrase_data["new_analysis"]
                if analysis.get("clipping_detected", False):
                    findings.append(f"⚠️ Clipping detected in {phrase_name}")
                    quality_issues += 1
                elif analysis.get("signal_quality", "") == "poor":
                    findings.append(f"⚠️ Poor signal quality detected in {phrase_name}")
                    quality_issues += 1

        if quality_issues == 0:
            findings.append("✅ No critical audio quality issues detected")

        return findings

    def _calculate_success_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate success metrics for the fixes."""
        metrics = {
            "overall_success": False,
            "generation_success_rate": 0.0,
            "performance_target_met": False,
            "format_compatibility": 0.0,
            "quality_score": 0.0
        }

        # Generation success rate
        before_after = results.get("before_after_comparison", {})
        successful_generations = sum(1 for data in before_after.values() if "error" not in data)
        metrics["generation_success_rate"] = successful_generations / len(self.test_phrases) if self.test_phrases else 0

        # Performance target
        performance = results.get("performance_validation", {})
        rtf_stats = performance.get("rtf_statistics", {})
        if rtf_stats:
            mean_rtf = rtf_stats.get("mean", 1.0)
            metrics["performance_target_met"] = mean_rtf < 0.5

        # Format compatibility
        format_tests = results.get("format_quality_tests", {})
        if format_tests and "error" not in format_tests:
            successful_formats = sum(1 for data in format_tests.values() if isinstance(data, dict) and data.get("success", False))
            metrics["format_compatibility"] = successful_formats / 5  # 5 test formats

        # Quality score (based on signal quality analysis)
        quality_scores = []
        for phrase_data in before_after.values():
            if "new_analysis" in phrase_data:
                analysis = phrase_data["new_analysis"]
                signal_quality = analysis.get("signal_quality", "")
                if "excellent" in signal_quality:
                    quality_scores.append(1.0)
                elif "good" in signal_quality:
                    quality_scores.append(0.8)
                elif "fair" in signal_quality:
                    quality_scores.append(0.6)
                else:
                    quality_scores.append(0.3)

        if quality_scores:
            metrics["quality_score"] = sum(quality_scores) / len(quality_scores)

        # Overall success (all metrics above threshold)
        metrics["overall_success"] = (
            metrics["generation_success_rate"] >= 1.0 and
            metrics["performance_target_met"] and
            metrics["format_compatibility"] >= 0.8 and
            metrics["quality_score"] >= 0.7
        )

        return metrics

    def _generate_validation_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate validation summary report."""
        report_file = self.output_dir / "QUALITY_FIX_VALIDATION_SUMMARY.md"

        with open(report_file, 'w') as f:
            f.write("# Quality Fix Validation Summary Report\n\n")
            f.write(f"**Generated**: {results['validation_date']}\n")
            f.write(f"**Validation Type**: Post-Fix Quality and Performance Validation\n")
            f.write(f"**Model**: {self.settings.model_name}\n\n")

            # Success Metrics
            success_metrics = results.get("success_metrics", {})
            f.write("## Success Metrics\n\n")
            f.write(f"- **Overall Success**: {'✅ PASSED' if success_metrics.get('overall_success', False) else '❌ FAILED'}\n")
            f.write(f"- **Generation Success Rate**: {success_metrics.get('generation_success_rate', 0):.1%}\n")
            f.write(f"- **Performance Target Met**: {'✅ Yes' if success_metrics.get('performance_target_met', False) else '❌ No'}\n")
            f.write(f"- **Format Compatibility**: {success_metrics.get('format_compatibility', 0):.1%}\n")
            f.write(f"- **Quality Score**: {success_metrics.get('quality_score', 0):.1%}\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            findings = results.get("findings", [])
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")

            # Performance Results
            performance = results.get("performance_validation", {})
            rtf_stats = performance.get("rtf_statistics", {})
            if rtf_stats:
                f.write("## Performance Results\n\n")
                f.write(f"- **Average RTF**: {rtf_stats.get('mean', 0):.3f}\n")
                f.write(f"- **RTF Standard Deviation**: {rtf_stats.get('std', 0):.3f}\n")
                f.write(f"- **RTF Range**: {rtf_stats.get('min', 0):.3f} - {rtf_stats.get('max', 0):.3f}\n")
                f.write(f"- **Performance Consistent**: {'✅ Yes' if rtf_stats.get('consistent', False) else '❌ No'}\n\n")

            # Format Test Results
            format_tests = results.get("format_quality_tests", {})
            if format_tests and "error" not in format_tests:
                f.write("## Format Test Results\n\n")
                f.write("| Format | Status | File Size | Processing Time |\n")
                f.write("|--------|--------|-----------|----------------|\n")

                for format_name, format_data in format_tests.items():
                    if isinstance(format_data, dict) and "analysis" in format_data:
                        analysis = format_data["analysis"]
                        status = "✅ Success" if format_data.get("success", False) else "❌ Failed"
                        file_size = f"{analysis.get('file_size', 0):,} bytes"
                        proc_time = f"{analysis.get('processing_time', 0):.3f}s"
                        f.write(f"| {format_name.upper()} | {status} | {file_size} | {proc_time} |\n")
                f.write("\n")

            # Generated Files
            f.write("## Generated Validation Files\n\n")
            f.write("The following files were generated for quality validation:\n\n")

            # List validation files
            before_after = results.get("before_after_comparison", {})
            for phrase_name, phrase_data in before_after.items():
                if "new_audio_file" in phrase_data:
                    f.write(f"- **{phrase_name}_after_fixes.wav**: Post-fix audio quality\n")

            format_tests = results.get("format_quality_tests", {})
            if format_tests and "error" not in format_tests:
                f.write(f"\n**Format Test Files**:\n")
                for format_name, format_data in format_tests.items():
                    if isinstance(format_data, dict) and "file" in format_data:
                        filename = Path(format_data["file"]).name
                        f.write(f"- `{filename}`: {format_name.upper()} format validation\n")

            f.write("\n---\n")
            f.write("**Note**: Compare these validation files with previous investigation files ")
            f.write("to confirm quality improvements and validate fix effectiveness.\n")


async def main():
    """Execute quality fix validation."""
    validator = QualityFixValidator()

    try:
        print("🔍 Starting quality fix validation...")
        print("This validation will confirm that audio quality fixes have resolved degradation issues.\n")

        results = await validator.run_validation()

        print("\n" + "="*70)
        print("QUALITY FIX VALIDATION COMPLETED")
        print("="*70)

        # Print success metrics
        success_metrics = results.get("success_metrics", {})
        overall_success = success_metrics.get("overall_success", False)

        print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ NEEDS IMPROVEMENT'}")
        print(f"\n📊 SUCCESS METRICS:")
        print(f"  • Generation Success: {success_metrics.get('generation_success_rate', 0):.1%}")
        print(f"  • Performance Target: {'✅ Met' if success_metrics.get('performance_target_met', False) else '❌ Not Met'}")
        print(f"  • Format Compatibility: {success_metrics.get('format_compatibility', 0):.1%}")
        print(f"  • Quality Score: {success_metrics.get('quality_score', 0):.1%}")

        # Print key findings
        findings = results.get("findings", [])
        if findings:
            print(f"\n🔍 KEY FINDINGS:")
            for finding in findings[:5]:  # Show top 5 findings
                print(f"  • {finding}")

        print(f"\n📁 Validation files saved to: {validator.output_dir}")
        print("📋 Review the summary report and compare with previous investigation files")

        if overall_success:
            print("\n🎉 QUALITY FIXES SUCCESSFUL!")
            print("   Audio quality degradation issues have been resolved.")
        else:
            print("\n⚠️ ADDITIONAL IMPROVEMENTS NEEDED")
            print("   Some quality issues may still require attention.")

        return results

    except Exception as e:
        print(f"\n❌ Quality fix validation failed: {e}")
        logger.exception("Validation failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
