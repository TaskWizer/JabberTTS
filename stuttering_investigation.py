#!/usr/bin/env python3
"""Stuttering Investigation Script for JabberTTS.

This script systematically investigates the root cause of stuttering artifacts
manifesting as "T-T-S" style fragmentation in generated speech.

Investigation Areas:
1. Audio Enhancement Pipeline Analysis
2. Raw Model Output Analysis  
3. Text Preprocessing Pipeline Investigation
4. Inference Pipeline Timing Analysis
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
from jabbertts.inference.preprocessing import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class StutteringInvestigator:
    """Systematic investigation of stuttering artifacts in JabberTTS."""
    
    def __init__(self):
        """Initialize the investigator."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.model_manager = get_model_manager()
        self.preprocessor = TextPreprocessor(use_phonemizer=True)
        
        # Create investigation output directory
        self.output_dir = Path("stuttering_investigation")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test cases specifically designed to trigger stuttering
        self.test_cases = [
            {
                "name": "simple_word",
                "text": "Welcome",
                "description": "Single word that was reported as intelligible"
            },
            {
                "name": "stuttering_trigger",
                "text": "Text-to-speech synthesis",
                "description": "Phrase that commonly triggers T-T-S stuttering"
            },
            {
                "name": "complex_phrase",
                "text": "The quick brown fox jumps over the lazy dog",
                "description": "Complex phrase to test syllable fragmentation"
            },
            {
                "name": "technical_terms",
                "text": "Neural network architecture optimization",
                "description": "Technical terms that may cause preprocessing issues"
            }
        ]
        
        print(f"Stuttering Investigation for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print(f"Model: {self.settings.model_name}")
        print()
    
    async def run_full_investigation(self) -> Dict[str, Any]:
        """Run the complete stuttering investigation."""
        print("=== STUTTERING INVESTIGATION STARTING ===\n")
        
        results = {
            "investigation_timestamp": time.time(),
            "test_cases": self.test_cases,
            "enhancement_analysis": {},
            "raw_model_analysis": {},
            "preprocessing_analysis": {},
            "timing_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        # Phase 1: Audio Enhancement Pipeline Analysis
        print("Phase 1: Audio Enhancement Pipeline Analysis")
        print("=" * 50)
        results["enhancement_analysis"] = await self._analyze_enhancement_pipeline()
        
        # Phase 2: Raw Model Output Analysis
        print("\nPhase 2: Raw Model Output Analysis")
        print("=" * 50)
        results["raw_model_analysis"] = await self._analyze_raw_model_output()
        
        # Phase 3: Text Preprocessing Analysis
        print("\nPhase 3: Text Preprocessing Pipeline Analysis")
        print("=" * 50)
        results["preprocessing_analysis"] = await self._analyze_preprocessing_pipeline()
        
        # Phase 4: Timing and Synchronization Analysis
        print("\nPhase 4: Inference Pipeline Timing Analysis")
        print("=" * 50)
        results["timing_analysis"] = await self._analyze_timing_issues()
        
        # Generate findings and recommendations
        results["findings"] = self._generate_findings(results)
        results["recommendations"] = self._generate_recommendations(results)
        
        # Save comprehensive results
        results_file = self.output_dir / "stuttering_investigation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        print(f"\n✓ Stuttering investigation completed!")
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Summary report: {self.output_dir}/STUTTERING_INVESTIGATION_SUMMARY.md")
        
        return results
    
    async def _analyze_enhancement_pipeline(self) -> Dict[str, Any]:
        """Analyze if audio enhancement is causing stuttering artifacts."""
        print("Investigating audio enhancement pipeline...")
        
        analysis_results = {
            "enhancement_disabled_samples": {},
            "enhancement_enabled_samples": {},
            "comparison_analysis": {},
            "enhancement_settings": {}
        }
        
        # Get current enhancement settings
        analysis_results["enhancement_settings"] = {
            "enable_audio_enhancement": self.settings.enable_audio_enhancement,
            "audio_quality": self.settings.audio_quality,
            "audio_normalization": self.settings.audio_normalization
        }
        
        print(f"Current enhancement settings: {analysis_results['enhancement_settings']}")
        
        for test_case in self.test_cases:
            print(f"\nTesting: {test_case['name']} - '{test_case['text']}'")
            
            # Test with enhancement DISABLED
            print("  Testing with enhancement DISABLED...")
            try:
                # Temporarily disable enhancement
                original_enhancement = self.settings.enable_audio_enhancement
                self.settings.enable_audio_enhancement = False
                
                result_disabled = await self.engine.generate_speech(
                    test_case["text"], voice="alloy"
                )
                
                # Save raw audio without enhancement
                audio_file_disabled = self.output_dir / f"{test_case['name']}_enhancement_disabled.wav"
                await self._save_audio_sample(
                    result_disabled["audio_data"], 
                    result_disabled["sample_rate"],
                    audio_file_disabled,
                    apply_enhancement=False
                )
                
                analysis_results["enhancement_disabled_samples"][test_case["name"]] = {
                    "audio_file": str(audio_file_disabled),
                    "duration": result_disabled["duration"],
                    "rtf": result_disabled["rtf"],
                    "sample_rate": result_disabled["sample_rate"]
                }
                
                print(f"    ✓ Disabled: RTF={result_disabled['rtf']:.3f}, Duration={result_disabled['duration']:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Enhancement disabled test failed: {e}")
                analysis_results["enhancement_disabled_samples"][test_case["name"]] = {"error": str(e)}
            finally:
                # Restore original setting
                self.settings.enable_audio_enhancement = original_enhancement
            
            # Test with enhancement ENABLED
            print("  Testing with enhancement ENABLED...")
            try:
                # Ensure enhancement is enabled
                self.settings.enable_audio_enhancement = True
                
                result_enabled = await self.engine.generate_speech(
                    test_case["text"], voice="alloy"
                )
                
                # Save audio with enhancement
                audio_file_enabled = self.output_dir / f"{test_case['name']}_enhancement_enabled.wav"
                await self._save_audio_sample(
                    result_enabled["audio_data"], 
                    result_enabled["sample_rate"],
                    audio_file_enabled,
                    apply_enhancement=True
                )
                
                analysis_results["enhancement_enabled_samples"][test_case["name"]] = {
                    "audio_file": str(audio_file_enabled),
                    "duration": result_enabled["duration"],
                    "rtf": result_enabled["rtf"],
                    "sample_rate": result_enabled["sample_rate"]
                }
                
                print(f"    ✓ Enabled: RTF={result_enabled['rtf']:.3f}, Duration={result_enabled['duration']:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Enhancement enabled test failed: {e}")
                analysis_results["enhancement_enabled_samples"][test_case["name"]] = {"error": str(e)}
        
        # Analyze differences
        analysis_results["comparison_analysis"] = self._compare_enhancement_results(
            analysis_results["enhancement_disabled_samples"],
            analysis_results["enhancement_enabled_samples"]
        )
        
        return analysis_results
    
    async def _analyze_raw_model_output(self) -> Dict[str, Any]:
        """Analyze raw SpeechT5 model output before any post-processing."""
        print("Analyzing raw model output...")
        
        analysis_results = {
            "raw_model_samples": {},
            "tensor_analysis": {},
            "model_info": {}
        }
        
        # Get model information
        model = self.model_manager.get_current_model()
        if not model or not model.is_loaded:
            model = await self.engine._ensure_model_loaded()
        
        analysis_results["model_info"] = {
            "model_name": self.settings.model_name,
            "device": str(model.device),
            "sample_rate": model.get_sample_rate(),
            "is_compiled": hasattr(model.model, '_orig_mod')  # Check if torch.compile was used
        }
        
        print(f"Model info: {analysis_results['model_info']}")
        
        for test_case in self.test_cases:
            print(f"\nAnalyzing raw output for: {test_case['name']}")
            
            try:
                # Generate raw model output without any post-processing
                raw_audio = await self._generate_raw_model_output(
                    model, test_case["text"]
                )
                
                # Save raw audio
                raw_file = self.output_dir / f"{test_case['name']}_raw_model_output.wav"
                await self._save_raw_audio(raw_audio, model.get_sample_rate(), raw_file)
                
                # Analyze tensor properties
                tensor_analysis = self._analyze_audio_tensor(raw_audio)
                
                analysis_results["raw_model_samples"][test_case["name"]] = {
                    "raw_audio_file": str(raw_file),
                    "tensor_shape": raw_audio.shape,
                    "tensor_dtype": str(raw_audio.dtype),
                    "duration": len(raw_audio) / model.get_sample_rate(),
                    "tensor_analysis": tensor_analysis
                }
                
                print(f"    ✓ Raw output: shape={raw_audio.shape}, duration={len(raw_audio) / model.get_sample_rate():.2f}s")
                
            except Exception as e:
                print(f"    ✗ Raw model analysis failed: {e}")
                analysis_results["raw_model_samples"][test_case["name"]] = {"error": str(e)}
        
        return analysis_results
    
    async def _generate_raw_model_output(self, model, text: str) -> np.ndarray:
        """Generate raw model output without any post-processing."""
        import torch
        
        # Preprocess text using the same pipeline
        processed_text = await self.engine._preprocess_text(text)
        
        # Use model directly to get raw output
        with torch.inference_mode():
            # Preprocess text
            inputs = model.processor(text=processed_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device, non_blocking=True)
            
            # Get speaker embeddings
            speaker_embeddings = model._get_speaker_embeddings("default").to(model.device, non_blocking=True)
            
            # Generate speech - RAW OUTPUT
            speech = model.model.generate_speech(
                input_ids,
                speaker_embeddings,
                vocoder=model.vocoder
            )
            
            # Convert to numpy WITHOUT any post-processing
            raw_audio = speech.detach().cpu().numpy()
        
        return raw_audio
    
    def _analyze_audio_tensor(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio tensor for potential stuttering indicators."""
        analysis = {
            "min_value": float(np.min(audio)),
            "max_value": float(np.max(audio)),
            "mean_value": float(np.mean(audio)),
            "std_value": float(np.std(audio)),
            "zero_crossings": int(np.sum(np.diff(np.sign(audio)) != 0)),
            "silence_ratio": float(np.sum(np.abs(audio) < 0.01) / len(audio)),
            "potential_clipping": bool(np.any(np.abs(audio) > 0.99)),
            "dynamic_range": float(np.max(audio) - np.min(audio))
        }
        
        # Detect potential stuttering patterns
        # Look for repetitive patterns or sudden amplitude changes
        if len(audio) > 1000:
            # Analyze amplitude envelope
            window_size = 160  # 10ms at 16kHz
            envelope = np.array([np.max(np.abs(audio[i:i+window_size])) 
                               for i in range(0, len(audio)-window_size, window_size)])
            
            # Look for repetitive patterns in envelope
            if len(envelope) > 10:
                envelope_diff = np.diff(envelope)
                analysis["envelope_variance"] = float(np.var(envelope))
                analysis["envelope_sudden_changes"] = int(np.sum(np.abs(envelope_diff) > 0.1))
                analysis["potential_stuttering_indicators"] = analysis["envelope_sudden_changes"] > len(envelope) * 0.1
        
        return analysis

    async def _analyze_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Analyze text preprocessing pipeline for fragmentation issues."""
        print("Analyzing text preprocessing pipeline...")

        analysis_results = {
            "phonemization_analysis": {},
            "text_normalization_analysis": {},
            "preprocessing_comparison": {}
        }

        for test_case in self.test_cases:
            print(f"\nAnalyzing preprocessing for: {test_case['name']}")

            try:
                original_text = test_case["text"]

                # Test with phonemization ENABLED
                print("  Testing with phonemization ENABLED...")
                self.preprocessor.use_phonemizer = True
                processed_with_phonemes = self.preprocessor.preprocess(original_text)

                # Test with phonemization DISABLED
                print("  Testing with phonemization DISABLED...")
                self.preprocessor.use_phonemizer = False
                processed_without_phonemes = self.preprocessor.preprocess(original_text)

                # Restore phonemization setting
                self.preprocessor.use_phonemizer = True

                analysis_results["preprocessing_comparison"][test_case["name"]] = {
                    "original_text": original_text,
                    "with_phonemes": processed_with_phonemes,
                    "without_phonemes": processed_without_phonemes,
                    "phoneme_length_change": len(processed_with_phonemes) - len(original_text),
                    "contains_phoneme_markers": any(char in processed_with_phonemes for char in ['ˈ', 'ˌ', 'ː']),
                    "potential_fragmentation": self._detect_text_fragmentation(processed_with_phonemes)
                }

                print(f"    ✓ Original: '{original_text}'")
                print(f"    ✓ With phonemes: '{processed_with_phonemes}'")
                print(f"    ✓ Without phonemes: '{processed_without_phonemes}'")

            except Exception as e:
                print(f"    ✗ Preprocessing analysis failed: {e}")
                analysis_results["preprocessing_comparison"][test_case["name"]] = {"error": str(e)}

        return analysis_results

    def _detect_text_fragmentation(self, text: str) -> Dict[str, Any]:
        """Detect potential text fragmentation that could cause stuttering."""
        fragmentation_indicators = {
            "excessive_spaces": text.count('  ') > 0,  # Double spaces
            "phoneme_boundaries": text.count(' ') > len(text.split()) * 2,  # Too many spaces
            "special_characters": sum(1 for char in text if not char.isalnum() and char not in ' .,!?'),
            "word_boundary_issues": any(word.strip() == '' for word in text.split()),
            "potential_syllable_separation": '-' in text or '_' in text
        }

        fragmentation_indicators["fragmentation_score"] = sum([
            fragmentation_indicators["excessive_spaces"],
            fragmentation_indicators["phoneme_boundaries"],
            fragmentation_indicators["special_characters"] > 5,
            fragmentation_indicators["word_boundary_issues"],
            fragmentation_indicators["potential_syllable_separation"]
        ])

        return fragmentation_indicators

    async def _analyze_timing_issues(self) -> Dict[str, Any]:
        """Analyze timing and synchronization issues in the inference pipeline."""
        print("Analyzing inference pipeline timing...")

        analysis_results = {
            "timing_measurements": {},
            "async_analysis": {},
            "memory_analysis": {}
        }

        for test_case in self.test_cases:
            print(f"\nTiming analysis for: {test_case['name']}")

            try:
                # Detailed timing measurement
                timing_data = await self._measure_detailed_timing(test_case["text"])
                analysis_results["timing_measurements"][test_case["name"]] = timing_data

                print(f"    ✓ Total time: {timing_data['total_time']:.3f}s")
                print(f"    ✓ Preprocessing: {timing_data['preprocessing_time']:.3f}s")
                print(f"    ✓ Model inference: {timing_data['inference_time']:.3f}s")
                print(f"    ✓ Post-processing: {timing_data['postprocessing_time']:.3f}s")

            except Exception as e:
                print(f"    ✗ Timing analysis failed: {e}")
                analysis_results["timing_measurements"][test_case["name"]] = {"error": str(e)}

        return analysis_results

    async def _measure_detailed_timing(self, text: str) -> Dict[str, float]:
        """Measure detailed timing of each pipeline stage."""
        timing_data = {}

        # Total timing
        total_start = time.time()

        # Preprocessing timing
        prep_start = time.time()
        processed_text = await self.engine._preprocess_text(text)
        timing_data["preprocessing_time"] = time.time() - prep_start

        # Model loading timing
        load_start = time.time()
        model = await self.engine._ensure_model_loaded()
        timing_data["model_loading_time"] = time.time() - load_start

        # Inference timing
        inf_start = time.time()
        audio_data = await self.engine._generate_audio(model, processed_text, "alloy", 1.0)
        timing_data["inference_time"] = time.time() - inf_start

        # Post-processing timing (audio processing)
        post_start = time.time()
        _, _ = await self.audio_processor.process_audio(
            audio_data, model.get_sample_rate(), "wav"
        )
        timing_data["postprocessing_time"] = time.time() - post_start

        timing_data["total_time"] = time.time() - total_start

        # Calculate RTF
        audio_duration = len(audio_data) / model.get_sample_rate()
        timing_data["audio_duration"] = audio_duration
        timing_data["rtf"] = timing_data["total_time"] / audio_duration if audio_duration > 0 else 0

        return timing_data

    async def _save_audio_sample(self, audio_data: np.ndarray, sample_rate: int,
                                filename: Path, apply_enhancement: bool = True) -> None:
        """Save audio sample with or without enhancement."""
        if apply_enhancement:
            # Use audio processor with enhancement
            audio_bytes, _ = await self.audio_processor.process_audio(
                audio_data, sample_rate, "wav"
            )
            with open(filename, 'wb') as f:
                f.write(audio_bytes)
        else:
            # Save raw audio without enhancement
            await self._save_raw_audio(audio_data, sample_rate, filename)

    async def _save_raw_audio(self, audio_data: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Save raw audio data as WAV without any processing."""
        try:
            import soundfile as sf
            sf.write(str(filename), audio_data, sample_rate)
        except ImportError:
            # Fallback to simple WAV creation
            self._create_simple_wav(audio_data, sample_rate, filename)

    def _create_simple_wav(self, audio: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Create a simple WAV file without external dependencies."""
        import struct
        import wave

        # Ensure audio is in correct format
        if audio.dtype != np.int16:
            # Convert float32 to int16
            audio = (audio * 32767).astype(np.int16)

        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

    def _compare_enhancement_results(self, disabled_results: Dict, enabled_results: Dict) -> Dict[str, Any]:
        """Compare results between enhancement disabled and enabled."""
        comparison = {
            "rtf_differences": {},
            "duration_differences": {},
            "enhancement_impact": {}
        }

        for test_name in disabled_results.keys():
            if test_name in enabled_results and "error" not in disabled_results[test_name] and "error" not in enabled_results[test_name]:
                disabled = disabled_results[test_name]
                enabled = enabled_results[test_name]

                rtf_diff = enabled["rtf"] - disabled["rtf"]
                duration_diff = enabled["duration"] - disabled["duration"]

                comparison["rtf_differences"][test_name] = {
                    "disabled_rtf": disabled["rtf"],
                    "enabled_rtf": enabled["rtf"],
                    "difference": rtf_diff,
                    "percentage_change": (rtf_diff / disabled["rtf"]) * 100 if disabled["rtf"] > 0 else 0
                }

                comparison["duration_differences"][test_name] = {
                    "disabled_duration": disabled["duration"],
                    "enabled_duration": enabled["duration"],
                    "difference": duration_diff
                }

        # Overall impact assessment
        rtf_diffs = [comp["difference"] for comp in comparison["rtf_differences"].values()]
        if rtf_diffs:
            comparison["enhancement_impact"] = {
                "average_rtf_increase": sum(rtf_diffs) / len(rtf_diffs),
                "max_rtf_increase": max(rtf_diffs),
                "enhancement_degrades_performance": sum(rtf_diffs) > 0,
                "significant_impact": any(abs(diff) > 0.1 for diff in rtf_diffs)
            }

        return comparison

    def _generate_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate findings based on investigation results."""
        findings = []

        # Enhancement pipeline findings
        enhancement_analysis = results.get("enhancement_analysis", {})
        if "comparison_analysis" in enhancement_analysis:
            impact = enhancement_analysis["comparison_analysis"].get("enhancement_impact", {})
            if impact.get("enhancement_degrades_performance", False):
                findings.append(
                    f"CRITICAL: Audio enhancement pipeline increases RTF by average of "
                    f"{impact.get('average_rtf_increase', 0):.3f}, degrading performance"
                )
            if impact.get("significant_impact", False):
                findings.append(
                    "CRITICAL: Audio enhancement has significant impact on performance (>0.1 RTF difference)"
                )

        # Raw model analysis findings
        raw_analysis = results.get("raw_model_analysis", {})
        for test_name, test_data in raw_analysis.get("raw_model_samples", {}).items():
            if "tensor_analysis" in test_data:
                tensor_analysis = test_data["tensor_analysis"]
                if tensor_analysis.get("potential_stuttering_indicators", False):
                    findings.append(
                        f"WARNING: Potential stuttering indicators detected in raw model output for '{test_name}'"
                    )
                if tensor_analysis.get("potential_clipping", False):
                    findings.append(
                        f"WARNING: Potential audio clipping detected in raw model output for '{test_name}'"
                    )

        # Preprocessing findings
        preprocessing_analysis = results.get("preprocessing_analysis", {})
        for test_name, test_data in preprocessing_analysis.get("preprocessing_comparison", {}).items():
            if "potential_fragmentation" in test_data:
                frag = test_data["potential_fragmentation"]
                if frag.get("fragmentation_score", 0) > 2:
                    findings.append(
                        f"WARNING: High text fragmentation score ({frag['fragmentation_score']}) "
                        f"detected for '{test_name}' - may cause stuttering"
                    )

        # Timing findings
        timing_analysis = results.get("timing_analysis", {})
        for test_name, timing_data in timing_analysis.get("timing_measurements", {}).items():
            if "rtf" in timing_data and timing_data["rtf"] > 0.5:
                findings.append(
                    f"PERFORMANCE: RTF target not met for '{test_name}' - "
                    f"RTF: {timing_data['rtf']:.3f} (target: <0.5)"
                )

        if not findings:
            findings.append("No critical issues detected in automated analysis - manual audio review required")

        return findings

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on investigation results."""
        recommendations = []

        # Enhancement-based recommendations
        enhancement_analysis = results.get("enhancement_analysis", {})
        if "comparison_analysis" in enhancement_analysis:
            impact = enhancement_analysis["comparison_analysis"].get("enhancement_impact", {})
            if impact.get("enhancement_degrades_performance", False):
                recommendations.append(
                    "IMMEDIATE: Disable audio enhancement by default and test if stuttering is resolved"
                )
                recommendations.append(
                    "INVESTIGATE: Review audio enhancement algorithms for potential stuttering-causing artifacts"
                )

        # Model-based recommendations
        raw_analysis = results.get("raw_model_analysis", {})
        model_info = raw_analysis.get("model_info", {})
        if model_info.get("is_compiled", False):
            recommendations.append(
                "TEST: Disable torch.compile optimization and test if stuttering is resolved"
            )

        # Preprocessing recommendations
        preprocessing_analysis = results.get("preprocessing_analysis", {})
        high_fragmentation_cases = []
        for test_name, test_data in preprocessing_analysis.get("preprocessing_comparison", {}).items():
            if "potential_fragmentation" in test_data:
                frag = test_data["potential_fragmentation"]
                if frag.get("fragmentation_score", 0) > 2:
                    high_fragmentation_cases.append(test_name)

        if high_fragmentation_cases:
            recommendations.append(
                f"INVESTIGATE: Review phonemization for cases with high fragmentation: {', '.join(high_fragmentation_cases)}"
            )
            recommendations.append(
                "TEST: Disable phonemization and test if stuttering is resolved"
            )

        # Performance recommendations
        timing_analysis = results.get("timing_analysis", {})
        slow_cases = []
        for test_name, timing_data in timing_analysis.get("timing_measurements", {}).items():
            if "rtf" in timing_data and timing_data["rtf"] > 0.5:
                slow_cases.append((test_name, timing_data["rtf"]))

        if slow_cases:
            recommendations.append(
                "OPTIMIZE: Implement ONNX Runtime conversion to improve performance"
            )
            recommendations.append(
                "OPTIMIZE: Apply model quantization (INT8/FP16) to reduce inference time"
            )

        # General recommendations
        recommendations.extend([
            "VALIDATE: Generate new human listening test samples with identified fixes",
            "MONITOR: Implement real-time stuttering detection in production",
            "DOCUMENT: Update troubleshooting guide with stuttering investigation findings"
        ])

        return recommendations

    def _generate_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive summary report."""
        report_file = self.output_dir / "STUTTERING_INVESTIGATION_SUMMARY.md"

        with open(report_file, 'w') as f:
            f.write("# Stuttering Investigation Summary Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Investigation Type**: Systematic Stuttering Root Cause Analysis\n")
            f.write(f"**Model**: {self.settings.model_name}\n")
            f.write(f"**Audio Quality**: {self.settings.audio_quality}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This investigation systematically analyzed the root causes of stuttering artifacts ")
            f.write("manifesting as 'T-T-S' style fragmentation in JabberTTS generated speech.\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            findings = results.get("findings", [])
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")

            # Recommendations
            f.write("## Immediate Recommendations\n\n")
            recommendations = results.get("recommendations", [])
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Enhancement Analysis Summary
            enhancement_analysis = results.get("enhancement_analysis", {})
            if "comparison_analysis" in enhancement_analysis:
                f.write("## Audio Enhancement Impact Analysis\n\n")
                impact = enhancement_analysis["comparison_analysis"].get("enhancement_impact", {})
                f.write(f"- **Average RTF Increase**: {impact.get('average_rtf_increase', 0):.3f}\n")
                f.write(f"- **Performance Degradation**: {impact.get('enhancement_degrades_performance', False)}\n")
                f.write(f"- **Significant Impact**: {impact.get('significant_impact', False)}\n\n")

            # Performance Summary
            timing_analysis = results.get("timing_analysis", {})
            if "timing_measurements" in timing_analysis:
                f.write("## Performance Analysis Summary\n\n")
                f.write("| Test Case | RTF | Target Met | Preprocessing | Inference | Post-processing |\n")
                f.write("|-----------|-----|------------|---------------|-----------|----------------|\n")

                for test_name, timing_data in timing_analysis["timing_measurements"].items():
                    if "error" not in timing_data:
                        rtf = timing_data.get("rtf", 0)
                        target_met = "✅" if rtf < 0.5 else "❌"
                        f.write(f"| {test_name} | {rtf:.3f} | {target_met} | ")
                        f.write(f"{timing_data.get('preprocessing_time', 0):.3f}s | ")
                        f.write(f"{timing_data.get('inference_time', 0):.3f}s | ")
                        f.write(f"{timing_data.get('postprocessing_time', 0):.3f}s |\n")
                f.write("\n")

            # Test Files Generated
            f.write("## Generated Test Files\n\n")
            f.write("The following audio samples were generated for analysis:\n\n")

            # Enhancement comparison files
            enhancement_analysis = results.get("enhancement_analysis", {})
            if "enhancement_disabled_samples" in enhancement_analysis:
                f.write("### Enhancement Comparison Samples\n")
                for test_name in enhancement_analysis["enhancement_disabled_samples"].keys():
                    f.write(f"- `{test_name}_enhancement_disabled.wav`\n")
                    f.write(f"- `{test_name}_enhancement_enabled.wav`\n")
                f.write("\n")

            # Raw model output files
            raw_analysis = results.get("raw_model_analysis", {})
            if "raw_model_samples" in raw_analysis:
                f.write("### Raw Model Output Samples\n")
                for test_name in raw_analysis["raw_model_samples"].keys():
                    f.write(f"- `{test_name}_raw_model_output.wav`\n")
                f.write("\n")

            f.write("## Next Steps\n\n")
            f.write("1. **Manual Audio Review**: Listen to generated samples to confirm automated findings\n")
            f.write("2. **Implement Priority Fixes**: Address highest priority issues first\n")
            f.write("3. **Validation Testing**: Generate new human listening test samples after fixes\n")
            f.write("4. **Performance Optimization**: Address RTF performance issues\n")
            f.write("5. **Regression Testing**: Ensure fixes don't introduce new issues\n\n")

            f.write("---\n")
            f.write("**Note**: This is an automated analysis. Manual audio review is required to confirm findings.\n")


async def main():
    """Run the stuttering investigation."""
    investigator = StutteringInvestigator()

    try:
        results = await investigator.run_full_investigation()

        print("\n" + "="*60)
        print("STUTTERING INVESTIGATION COMPLETED")
        print("="*60)

        # Print key findings
        findings = results.get("findings", [])
        if findings:
            print("\n🔍 KEY FINDINGS:")
            for finding in findings[:5]:  # Show top 5 findings
                print(f"  • {finding}")

        # Print top recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                print(f"  • {rec}")

        print(f"\n📁 Investigation files saved to: {investigator.output_dir}")
        print("📋 Review the summary report for detailed analysis")

        return results

    except Exception as e:
        print(f"\n❌ Stuttering investigation failed: {e}")
        logger.exception("Investigation failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
