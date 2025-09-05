#!/usr/bin/env python3
"""Phonemization Stuttering Test for JabberTTS.

This script specifically tests whether eSpeak-NG phonemization is causing
stuttering artifacts by comparing phonemized vs non-phonemized text processing.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhonemizationStutteringTest:
    """Test phonemization impact on stuttering artifacts."""
    
    def __init__(self):
        """Initialize the test."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.model_manager = get_model_manager()
        
        # Create test output directory
        self.output_dir = Path("phonemization_stuttering_test")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test cases specifically for phonemization analysis
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
                "name": "complex_consonants",
                "text": "Strength through struggle",
                "description": "Complex consonant clusters that may cause phonemization issues"
            },
            {
                "name": "technical_terms",
                "text": "Neural network architecture optimization",
                "description": "Technical terms with complex phonemization"
            },
            {
                "name": "stuttering_test",
                "text": "Testing text-to-speech stuttering artifacts",
                "description": "Full test phrase for stuttering detection"
            }
        ]
        
        print(f"Phonemization Stuttering Test for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print()
    
    async def run_phonemization_comparison_test(self) -> Dict[str, Any]:
        """Run comparison test between phonemized and non-phonemized text processing."""
        print("=== PHONEMIZATION STUTTERING TEST ===\n")
        
        results = {
            "test_timestamp": time.time(),
            "phonemized_results": {},
            "non_phonemized_results": {},
            "phonemization_analysis": {},
            "comparison_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        # Phase 1: Test with phonemization ENABLED (current state)
        print("Phase 1: Testing with phonemization ENABLED")
        print("=" * 50)
        results["phonemized_results"] = await self._test_with_phonemization_enabled()
        
        # Phase 2: Test with phonemization DISABLED
        print("\nPhase 2: Testing with phonemization DISABLED")
        print("=" * 50)
        results["non_phonemized_results"] = await self._test_with_phonemization_disabled()
        
        # Phase 3: Analyze phonemization patterns
        print("\nPhase 3: Analyzing Phonemization Patterns")
        print("=" * 50)
        results["phonemization_analysis"] = await self._analyze_phonemization_patterns()
        
        # Phase 4: Compare results
        print("\nPhase 4: Comparing Results")
        print("=" * 50)
        results["comparison_analysis"] = self._compare_phonemization_results(
            results["phonemized_results"],
            results["non_phonemized_results"]
        )
        
        # Generate findings and recommendations
        results["findings"] = self._generate_phonemization_findings(results)
        results["recommendations"] = self._generate_phonemization_recommendations(results)
        
        # Save results
        results_file = self.output_dir / "phonemization_stuttering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_phonemization_summary_report(results)
        
        print(f"\n✓ Phonemization stuttering test completed!")
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Summary report: {self.output_dir}/PHONEMIZATION_STUTTERING_SUMMARY.md")
        
        return results
    
    async def _test_with_phonemization_enabled(self) -> Dict[str, Any]:
        """Test with phonemization enabled (current state)."""
        print("Testing with phonemization ENABLED...")
        
        results = {
            "phonemization_status": True,
            "test_results": {}
        }
        
        for test_case in self.test_cases:
            print(f"\nTesting: {test_case['name']} - '{test_case['text']}'")
            
            try:
                # Generate speech with phonemization enabled
                start_time = time.time()
                result = await self.engine.generate_speech(test_case["text"], voice="alloy")
                generation_time = time.time() - start_time
                
                # Save audio sample
                audio_file = self.output_dir / f"{test_case['name']}_phonemized.wav"
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
                
                print(f"    ✓ Phonemized: RTF={result['rtf']:.3f}, Duration={result['duration']:.2f}s")
                print(f"    ✓ Stuttering score: {stuttering_analysis['stuttering_score']:.3f}")
                
            except Exception as e:
                print(f"    ✗ Phonemized test failed: {e}")
                results["test_results"][test_case["name"]] = {"error": str(e)}
        
        return results
    
    async def _test_with_phonemization_disabled(self) -> Dict[str, Any]:
        """Test with phonemization disabled."""
        print("Testing with phonemization DISABLED...")
        
        # Temporarily disable phonemization in the engine
        original_use_phonemizer = self.engine.preprocessor.use_phonemizer
        self.engine.preprocessor.use_phonemizer = False
        
        try:
            results = {
                "phonemization_status": False,
                "test_results": {}
            }
            
            for test_case in self.test_cases:
                print(f"\nTesting: {test_case['name']} - '{test_case['text']}'")
                
                try:
                    # Generate speech with phonemization disabled
                    start_time = time.time()
                    result = await self.engine.generate_speech(test_case["text"], voice="alloy")
                    generation_time = time.time() - start_time
                    
                    # Save audio sample
                    audio_file = self.output_dir / f"{test_case['name']}_non_phonemized.wav"
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
                    
                    print(f"    ✓ Non-phonemized: RTF={result['rtf']:.3f}, Duration={result['duration']:.2f}s")
                    print(f"    ✓ Stuttering score: {stuttering_analysis['stuttering_score']:.3f}")
                    
                except Exception as e:
                    print(f"    ✗ Non-phonemized test failed: {e}")
                    results["test_results"][test_case["name"]] = {"error": str(e)}
            
        finally:
            # Restore original phonemization setting
            self.engine.preprocessor.use_phonemizer = original_use_phonemizer
        
        return results
    
    async def _analyze_phonemization_patterns(self) -> Dict[str, Any]:
        """Analyze phonemization patterns for potential stuttering causes."""
        print("Analyzing phonemization patterns...")
        
        preprocessor_with_phonemes = TextPreprocessor(use_phonemizer=True)
        preprocessor_without_phonemes = TextPreprocessor(use_phonemizer=False)
        
        analysis_results = {
            "phonemization_comparison": {},
            "fragmentation_analysis": {},
            "phoneme_complexity": {}
        }
        
        for test_case in self.test_cases:
            print(f"\nAnalyzing phonemization for: {test_case['name']}")
            
            try:
                original_text = test_case["text"]
                
                # Process with phonemization
                phonemized_text = preprocessor_with_phonemes.preprocess(original_text)
                
                # Process without phonemization
                non_phonemized_text = preprocessor_without_phonemes.preprocess(original_text)
                
                # Analyze differences
                phoneme_analysis = self._analyze_phoneme_complexity(original_text, phonemized_text)
                fragmentation_analysis = self._analyze_text_fragmentation(phonemized_text)
                
                analysis_results["phonemization_comparison"][test_case["name"]] = {
                    "original_text": original_text,
                    "phonemized_text": phonemized_text,
                    "non_phonemized_text": non_phonemized_text,
                    "length_change": len(phonemized_text) - len(original_text),
                    "word_count_change": len(phonemized_text.split()) - len(original_text.split()),
                    "phoneme_analysis": phoneme_analysis,
                    "fragmentation_analysis": fragmentation_analysis
                }
                
                print(f"    ✓ Original: '{original_text}'")
                print(f"    ✓ Phonemized: '{phonemized_text}'")
                print(f"    ✓ Fragmentation score: {fragmentation_analysis['fragmentation_score']}")
                
            except Exception as e:
                print(f"    ✗ Phonemization analysis failed: {e}")
                analysis_results["phonemization_comparison"][test_case["name"]] = {"error": str(e)}
        
        return analysis_results

    def _analyze_phoneme_complexity(self, original_text: str, phonemized_text: str) -> Dict[str, Any]:
        """Analyze how eSpeak-NG phonemization transforms text and identify potential stuttering triggers."""
        analysis = {
            "phoneme_markers": {
                "primary_stress": phonemized_text.count('ˈ'),
                "secondary_stress": phonemized_text.count('ˌ'),
                "long_vowels": phonemized_text.count('ː'),
                "total_markers": phonemized_text.count('ˈ') + phonemized_text.count('ˌ') + phonemized_text.count('ː')
            },
            "syllable_analysis": {},
            "consonant_clusters": {},
            "phoneme_density": {}
        }

        # Syllable boundary analysis
        original_syllables = len(original_text.split())  # Rough estimate
        phoneme_words = phonemized_text.split()
        phonemized_syllables = len(phoneme_words)

        analysis["syllable_analysis"] = {
            "original_word_count": original_syllables,
            "phonemized_word_count": phonemized_syllables,
            "syllable_boundary_change": phonemized_syllables - original_syllables,
            "syllable_fragmentation": phonemized_syllables > original_syllables * 1.5
        }

        # Complex consonant cluster detection
        consonant_clusters = []
        cluster_pattern = r'[bcdfghjklmnpqrstvwxyz]{3,}'  # 3+ consecutive consonants
        import re
        clusters = re.findall(cluster_pattern, phonemized_text.lower())

        analysis["consonant_clusters"] = {
            "cluster_count": len(clusters),
            "clusters_found": clusters,
            "max_cluster_length": max(len(cluster) for cluster in clusters) if clusters else 0,
            "complex_clusters": len([c for c in clusters if len(c) >= 4])
        }

        # Phoneme density analysis
        phoneme_chars = sum(1 for char in phonemized_text if char.isalpha() or char in 'ˈˌː')
        text_length = len(phonemized_text.replace(' ', ''))

        analysis["phoneme_density"] = {
            "phoneme_char_count": phoneme_chars,
            "total_char_count": text_length,
            "phoneme_density_ratio": phoneme_chars / text_length if text_length > 0 else 0,
            "marker_density": analysis["phoneme_markers"]["total_markers"] / len(original_text) if len(original_text) > 0 else 0
        }

        # Calculate complexity score
        complexity_score = (
            analysis["phoneme_markers"]["total_markers"] * 0.3 +
            analysis["consonant_clusters"]["complex_clusters"] * 0.4 +
            (1 if analysis["syllable_analysis"]["syllable_fragmentation"] else 0) * 0.3
        )

        analysis["complexity_score"] = complexity_score
        analysis["high_complexity"] = complexity_score > 2.0

        return analysis

    def _analyze_text_fragmentation(self, text: str) -> Dict[str, Any]:
        """Detect text fragmentation patterns that could cause syllable separation."""
        fragmentation_indicators = {
            "excessive_spaces": text.count('  ') > 0,  # Double spaces
            "phoneme_boundaries": text.count(' ') > len(text.split()) * 2,  # Too many spaces
            "special_characters": sum(1 for char in text if not char.isalnum() and char not in ' .,!?ˈˌː'),
            "word_boundary_issues": any(word.strip() == '' for word in text.split()),
            "potential_syllable_separation": '-' in text or '_' in text,
            "phoneme_marker_density": (text.count('ˈ') + text.count('ˌ') + text.count('ː')) / len(text) if len(text) > 0 else 0
        }

        # Analyze space distribution
        words = text.split()
        if len(words) > 1:
            space_positions = [i for i, char in enumerate(text) if char == ' ']
            space_intervals = [space_positions[i+1] - space_positions[i] for i in range(len(space_positions)-1)]

            fragmentation_indicators["space_analysis"] = {
                "total_spaces": len(space_positions),
                "average_space_interval": sum(space_intervals) / len(space_intervals) if space_intervals else 0,
                "irregular_spacing": len([interval for interval in space_intervals if interval < 3]) > 0
            }
        else:
            fragmentation_indicators["space_analysis"] = {
                "total_spaces": 0,
                "average_space_interval": 0,
                "irregular_spacing": False
            }

        # Calculate fragmentation score
        fragmentation_score = sum([
            fragmentation_indicators["excessive_spaces"] * 2,
            fragmentation_indicators["phoneme_boundaries"] * 3,
            min(fragmentation_indicators["special_characters"] / 5, 2),  # Cap at 2
            fragmentation_indicators["word_boundary_issues"] * 2,
            fragmentation_indicators["potential_syllable_separation"] * 1,
            fragmentation_indicators["phoneme_marker_density"] * 10,  # High weight for phoneme markers
            fragmentation_indicators["space_analysis"]["irregular_spacing"] * 1
        ])

        fragmentation_indicators["fragmentation_score"] = fragmentation_score
        fragmentation_indicators["high_fragmentation"] = fragmentation_score > 3.0

        return fragmentation_indicators

    def _analyze_stuttering_patterns(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio for stuttering artifacts using amplitude envelope analysis."""
        analysis = {
            "stuttering_score": 0.0,
            "repetitive_patterns": 0,
            "sudden_amplitude_changes": 0,
            "silence_gaps": 0,
            "potential_stuttering": False,
            "envelope_analysis": {}
        }

        if len(audio) < 1000:
            return analysis

        # Compute amplitude envelope with 10ms windows (160 samples at 16kHz)
        window_size = 160  # 10ms at 16kHz
        envelope = np.array([np.max(np.abs(audio[i:i+window_size]))
                           for i in range(0, len(audio)-window_size, window_size)])

        if len(envelope) < 10:
            return analysis

        # Detect sudden amplitude changes (>0.1 threshold)
        envelope_diff = np.diff(envelope)
        sudden_changes = np.sum(np.abs(envelope_diff) > 0.1)
        analysis["sudden_amplitude_changes"] = int(sudden_changes)

        # Identify repetitive patterns using correlation analysis
        pattern_length = 5  # 50ms patterns
        repetitive_count = 0

        for i in range(len(envelope) - pattern_length * 2):
            pattern1 = envelope[i:i+pattern_length]
            pattern2 = envelope[i+pattern_length:i+pattern_length*2]

            # Calculate correlation for patterns with sufficient variance
            if np.std(pattern1) > 0.01 and np.std(pattern2) > 0.01:
                correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                if correlation > 0.8:  # High correlation indicates repetition
                    repetitive_count += 1

        analysis["repetitive_patterns"] = repetitive_count

        # Count silence gaps that indicate stuttering
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

        # Envelope analysis details
        analysis["envelope_analysis"] = {
            "envelope_length": len(envelope),
            "envelope_mean": float(np.mean(envelope)),
            "envelope_std": float(np.std(envelope)),
            "envelope_max": float(np.max(envelope)),
            "envelope_min": float(np.min(envelope)),
            "dynamic_range": float(np.max(envelope) - np.min(envelope))
        }

        # Calculate overall stuttering score (0.0-1.0 scale)
        stuttering_score = (
            (sudden_changes / len(envelope)) * 0.4 +
            (repetitive_count / max(len(envelope) - pattern_length * 2, 1)) * 0.4 +
            (silence_gaps / max(len(envelope) // 10, 1)) * 0.2
        )

        analysis["stuttering_score"] = float(min(stuttering_score, 1.0))  # Cap at 1.0
        analysis["potential_stuttering"] = stuttering_score > 0.1

        return analysis

    async def _save_audio_sample(self, audio_data: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Save audio samples using the existing AudioProcessor for consistent formatting."""
        audio_bytes, _ = await self.audio_processor.process_audio(
            audio_data, sample_rate, "wav"
        )
        with open(filename, 'wb') as f:
            f.write(audio_bytes)

    def _compare_phonemization_results(self, phonemized_results: Dict, non_phonemized_results: Dict) -> Dict[str, Any]:
        """Compare results between phonemized and non-phonemized processing."""
        comparison = {
            "performance_differences": {},
            "stuttering_differences": {},
            "phonemization_impact": {}
        }

        for test_name in phonemized_results.get("test_results", {}):
            if test_name in non_phonemized_results.get("test_results", {}):
                phonemized = phonemized_results["test_results"][test_name]
                non_phonemized = non_phonemized_results["test_results"][test_name]

                if "error" not in phonemized and "error" not in non_phonemized:
                    # Performance comparison
                    rtf_diff = phonemized["rtf"] - non_phonemized["rtf"]
                    duration_diff = phonemized["duration"] - non_phonemized["duration"]

                    comparison["performance_differences"][test_name] = {
                        "phonemized_rtf": phonemized["rtf"],
                        "non_phonemized_rtf": non_phonemized["rtf"],
                        "rtf_difference": rtf_diff,
                        "rtf_improvement": rtf_diff < 0,
                        "phonemized_duration": phonemized["duration"],
                        "non_phonemized_duration": non_phonemized["duration"],
                        "duration_difference": duration_diff
                    }

                    # Stuttering comparison
                    phonemized_stuttering = phonemized["stuttering_analysis"]["stuttering_score"]
                    non_phonemized_stuttering = non_phonemized["stuttering_analysis"]["stuttering_score"]
                    stuttering_diff = phonemized_stuttering - non_phonemized_stuttering

                    comparison["stuttering_differences"][test_name] = {
                        "phonemized_stuttering_score": phonemized_stuttering,
                        "non_phonemized_stuttering_score": non_phonemized_stuttering,
                        "stuttering_difference": stuttering_diff,
                        "phonemization_increases_stuttering": stuttering_diff > 0.05,
                        "phonemized_potential_stuttering": phonemized["stuttering_analysis"]["potential_stuttering"],
                        "non_phonemized_potential_stuttering": non_phonemized["stuttering_analysis"]["potential_stuttering"]
                    }

        # Overall impact assessment
        rtf_diffs = [comp["rtf_difference"] for comp in comparison["performance_differences"].values()]
        stuttering_diffs = [comp["stuttering_difference"] for comp in comparison["stuttering_differences"].values()]

        if rtf_diffs and stuttering_diffs:
            comparison["phonemization_impact"] = {
                "average_rtf_change": sum(rtf_diffs) / len(rtf_diffs),
                "average_stuttering_change": sum(stuttering_diffs) / len(stuttering_diffs),
                "phonemization_improves_performance": sum(rtf_diffs) < 0,
                "phonemization_increases_stuttering": sum(stuttering_diffs) > 0,
                "significant_stuttering_impact": any(abs(diff) > 0.05 for diff in stuttering_diffs),
                "stuttering_reduction_cases": len([diff for diff in stuttering_diffs if diff < -0.05]),
                "stuttering_increase_cases": len([diff for diff in stuttering_diffs if diff > 0.05])
            }

        return comparison

    def _generate_phonemization_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate specific findings about phonemization impact on stuttering."""
        findings = []

        comparison = results.get("comparison_analysis", {})
        impact = comparison.get("phonemization_impact", {})

        if impact.get("phonemization_increases_stuttering", False):
            findings.append(
                f"CRITICAL: Phonemization increases stuttering artifacts by average of "
                f"{impact.get('average_stuttering_change', 0):.3f}"
            )

        if impact.get("significant_stuttering_impact", False):
            findings.append(
                "CRITICAL: Phonemization has significant impact on stuttering (>0.05 difference)"
            )

        # Check individual test cases
        for test_name, stuttering_data in comparison.get("stuttering_differences", {}).items():
            if stuttering_data.get("phonemization_increases_stuttering", False):
                findings.append(
                    f"WARNING: Phonemization increases stuttering for '{test_name}' - "
                    f"score: {stuttering_data['phonemized_stuttering_score']:.3f} vs "
                    f"{stuttering_data['non_phonemized_stuttering_score']:.3f}"
                )

        # Analyze phonemization patterns
        phonemization_analysis = results.get("phonemization_analysis", {})
        high_complexity_cases = []
        high_fragmentation_cases = []

        for test_name, test_data in phonemization_analysis.get("phonemization_comparison", {}).items():
            if "phoneme_analysis" in test_data and test_data["phoneme_analysis"].get("high_complexity", False):
                high_complexity_cases.append(test_name)
            if "fragmentation_analysis" in test_data and test_data["fragmentation_analysis"].get("high_fragmentation", False):
                high_fragmentation_cases.append(test_name)

        if high_complexity_cases:
            findings.append(
                f"WARNING: High phoneme complexity detected in: {', '.join(high_complexity_cases)}"
            )

        if high_fragmentation_cases:
            findings.append(
                f"WARNING: High text fragmentation detected in: {', '.join(high_fragmentation_cases)}"
            )

        if not findings:
            findings.append("Phonemization does not appear to be causing stuttering artifacts")

        return findings

    def _generate_phonemization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Provide actionable recommendations based on test results."""
        recommendations = []

        comparison = results.get("comparison_analysis", {})
        impact = comparison.get("phonemization_impact", {})

        if impact.get("phonemization_increases_stuttering", False):
            recommendations.append(
                "IMMEDIATE: Disable phonemization in production to eliminate stuttering"
            )
            recommendations.append(
                "INVESTIGATE: Review eSpeak-NG configuration for speech generation compatibility"
            )
        else:
            recommendations.append(
                "CONTINUE: Phonemization can be safely used - stuttering cause is elsewhere"
            )

        if impact.get("phonemization_improves_performance", True):
            recommendations.append(
                "OPTIMIZE: Phonemization provides performance benefits - keep enabled if no stuttering"
            )

        # Specific recommendations based on analysis
        phonemization_analysis = results.get("phonemization_analysis", {})
        for test_name, test_data in phonemization_analysis.get("phonemization_comparison", {}).items():
            if "phoneme_analysis" in test_data:
                phoneme_analysis = test_data["phoneme_analysis"]
                if phoneme_analysis.get("high_complexity", False):
                    recommendations.append(
                        f"INVESTIGATE: Review phoneme complexity for '{test_name}' - "
                        f"complexity score: {phoneme_analysis.get('complexity_score', 0):.2f}"
                    )

        recommendations.extend([
            "VALIDATE: Generate human listening test samples with phonemization disabled",
            "MONITOR: Implement phonemization quality checks in production pipeline",
            "DOCUMENT: Update text preprocessing guidelines based on findings",
            "TEST: Experiment with alternative phonemization backends if available"
        ])

        return recommendations

    def _generate_phonemization_summary_report(self, results: Dict[str, Any]) -> None:
        """Create comprehensive markdown report with test results table, audio file references, and conclusions."""
        report_file = self.output_dir / "PHONEMIZATION_STUTTERING_SUMMARY.md"

        with open(report_file, 'w') as f:
            f.write("# Phonemization Stuttering Analysis Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analysis Type**: eSpeak-NG Phonemization Impact on Stuttering Artifacts\n")
            f.write(f"**Model**: {self.settings.model_name}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This analysis specifically tested whether eSpeak-NG phonemization is causing ")
            f.write("stuttering artifacts by comparing phonemized vs non-phonemized text processing.\n\n")

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

            # Phonemization Impact Analysis
            comparison = results.get("comparison_analysis", {})
            impact = comparison.get("phonemization_impact", {})
            if impact:
                f.write("## Phonemization Impact Analysis\n\n")
                f.write(f"- **Average RTF Change**: {impact.get('average_rtf_change', 0):.3f}\n")
                f.write(f"- **Average Stuttering Change**: {impact.get('average_stuttering_change', 0):.3f}\n")
                f.write(f"- **Performance Improvement**: {impact.get('phonemization_improves_performance', False)}\n")
                f.write(f"- **Increases Stuttering**: {impact.get('phonemization_increases_stuttering', False)}\n")
                f.write(f"- **Stuttering Reduction Cases**: {impact.get('stuttering_reduction_cases', 0)}\n")
                f.write(f"- **Stuttering Increase Cases**: {impact.get('stuttering_increase_cases', 0)}\n\n")

            # Test Results Summary Table
            f.write("## Test Results Summary\n\n")
            f.write("| Test Case | Phonemized RTF | Non-Phonemized RTF | Phonemized Stuttering | Non-Phonemized Stuttering | Difference |\n")
            f.write("|-----------|----------------|--------------------|-----------------------|----------------------------|------------|\n")

            for test_name, perf_data in comparison.get("performance_differences", {}).items():
                stuttering_data = comparison.get("stuttering_differences", {}).get(test_name, {})
                stuttering_diff = stuttering_data.get("stuttering_difference", 0)
                diff_indicator = "⬆️" if stuttering_diff > 0.05 else "⬇️" if stuttering_diff < -0.05 else "➡️"

                f.write(f"| {test_name} | {perf_data.get('phonemized_rtf', 0):.3f} | ")
                f.write(f"{perf_data.get('non_phonemized_rtf', 0):.3f} | ")
                f.write(f"{stuttering_data.get('phonemized_stuttering_score', 0):.3f} | ")
                f.write(f"{stuttering_data.get('non_phonemized_stuttering_score', 0):.3f} | ")
                f.write(f"{diff_indicator} {stuttering_diff:+.3f} |\n")
            f.write("\n")

            # Phonemization Pattern Analysis
            phonemization_analysis = results.get("phonemization_analysis", {})
            if "phonemization_comparison" in phonemization_analysis:
                f.write("## Phonemization Pattern Analysis\n\n")
                f.write("| Test Case | Original Text | Phonemized Text | Complexity Score | Fragmentation Score |\n")
                f.write("|-----------|---------------|-----------------|------------------|--------------------|\n")

                for test_name, test_data in phonemization_analysis["phonemization_comparison"].items():
                    if "error" not in test_data:
                        original = test_data.get("original_text", "")
                        phonemized = test_data.get("phonemized_text", "")
                        complexity = test_data.get("phoneme_analysis", {}).get("complexity_score", 0)
                        fragmentation = test_data.get("fragmentation_analysis", {}).get("fragmentation_score", 0)

                        f.write(f"| {test_name} | {original} | {phonemized} | {complexity:.2f} | {fragmentation:.2f} |\n")
                f.write("\n")

            # Generated Audio Files
            f.write("## Generated Audio Files\n\n")
            f.write("Compare these audio files to identify phonemization impact on stuttering:\n\n")

            for test_case in self.test_cases:
                f.write(f"### {test_case['name'].replace('_', ' ').title()}\n")
                f.write(f"**Text**: \"{test_case['text']}\"\n")
                f.write(f"- Phonemized: `{test_case['name']}_phonemized.wav`\n")
                f.write(f"- Non-phonemized: `{test_case['name']}_non_phonemized.wav`\n\n")

            # Detailed Analysis
            f.write("## Detailed Phonemization Analysis\n\n")
            for test_name, test_data in phonemization_analysis.get("phonemization_comparison", {}).items():
                if "error" not in test_data:
                    f.write(f"### {test_name.replace('_', ' ').title()}\n")
                    f.write(f"**Original**: {test_data.get('original_text', '')}\n")
                    f.write(f"**Phonemized**: {test_data.get('phonemized_text', '')}\n")

                    phoneme_analysis = test_data.get("phoneme_analysis", {})
                    if phoneme_analysis:
                        f.write(f"**Phoneme Markers**: {phoneme_analysis.get('phoneme_markers', {}).get('total_markers', 0)}\n")
                        f.write(f"**Complexity Score**: {phoneme_analysis.get('complexity_score', 0):.2f}\n")
                        f.write(f"**High Complexity**: {phoneme_analysis.get('high_complexity', False)}\n")

                    fragmentation_analysis = test_data.get("fragmentation_analysis", {})
                    if fragmentation_analysis:
                        f.write(f"**Fragmentation Score**: {fragmentation_analysis.get('fragmentation_score', 0):.2f}\n")
                        f.write(f"**High Fragmentation**: {fragmentation_analysis.get('high_fragmentation', False)}\n")

                    f.write("\n")

            f.write("---\n")
            f.write("**Note**: Manual audio comparison is essential to confirm automated analysis.\n")
            f.write("**Legend**: ⬆️ = Stuttering increased, ⬇️ = Stuttering decreased, ➡️ = No significant change\n")


async def main():
    """Execute the complete phonemization stuttering test workflow with proper error handling and progress reporting."""
    test = PhonemizationStutteringTest()

    try:
        print("🔍 Starting comprehensive phonemization stuttering analysis...")
        print("This test will determine if eSpeak-NG phonemization is causing T-T-S stuttering artifacts.\n")

        results = await test.run_phonemization_comparison_test()

        print("\n" + "="*70)
        print("PHONEMIZATION STUTTERING TEST COMPLETED")
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

        # Print impact summary
        comparison = results.get("comparison_analysis", {})
        impact = comparison.get("phonemization_impact", {})
        if impact:
            print(f"\n📊 IMPACT SUMMARY:")
            print(f"  • Average stuttering change: {impact.get('average_stuttering_change', 0):+.3f}")
            print(f"  • Cases with reduced stuttering: {impact.get('stuttering_reduction_cases', 0)}")
            print(f"  • Cases with increased stuttering: {impact.get('stuttering_increase_cases', 0)}")
            print(f"  • Significant impact detected: {impact.get('significant_stuttering_impact', False)}")

        print(f"\n📁 Test files saved to: {test.output_dir}")
        print("📋 Review the summary report and listen to audio samples for manual validation")
        print("\n🎧 MANUAL LISTENING TEST REQUIRED:")
        print("   Listen to the generated audio files to confirm automated analysis")
        print("   Pay special attention to T-T-S style stuttering artifacts")

        return results

    except Exception as e:
        print(f"\n❌ Phonemization stuttering test failed: {e}")
        logger.exception("Test failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
