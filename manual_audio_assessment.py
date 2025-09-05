#!/usr/bin/env python3
"""Manual Audio Quality Assessment Tool for JabberTTS.

This tool provides a systematic framework for human listening analysis of
generated audio samples to identify quality degradation patterns and rank
samples by intelligibility and naturalness.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import sys


class ManualAudioAssessment:
    """Tool for systematic manual audio quality assessment."""
    
    def __init__(self):
        """Initialize the assessment tool."""
        self.assessment_data = {
            "assessment_timestamp": time.time(),
            "assessment_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_samples": 0,
            "samples": {},
            "summary_statistics": {},
            "quality_rankings": {},
            "findings": []
        }
        
        # Define all audio samples to assess
        self.audio_samples = [
            # Inference Timing Test Samples
            {"path": "./inference_timing_stuttering_test/sequential_test_1_sequential.wav", "text": "Welcome to the system", "category": "timing_analysis"},
            {"path": "./inference_timing_stuttering_test/sequential_test_2_sequential.wav", "text": "Text-to-speech synthesis", "category": "timing_analysis"},
            {"path": "./inference_timing_stuttering_test/sequential_test_3_sequential.wav", "text": "Testing stuttering artifacts", "category": "timing_analysis"},
            
            # Phonemization Test Samples
            {"path": "./phonemization_stuttering_test/welcome_word_phonemized.wav", "text": "Welcome", "category": "phonemization_enabled"},
            {"path": "./phonemization_stuttering_test/welcome_word_non_phonemized.wav", "text": "Welcome", "category": "phonemization_disabled"},
            {"path": "./phonemization_stuttering_test/tts_phrase_phonemized.wav", "text": "Text-to-speech synthesis", "category": "phonemization_enabled"},
            {"path": "./phonemization_stuttering_test/tts_phrase_non_phonemized.wav", "text": "Text-to-speech synthesis", "category": "phonemization_disabled"},
            {"path": "./phonemization_stuttering_test/complex_consonants_phonemized.wav", "text": "Strength through struggle", "category": "phonemization_enabled"},
            {"path": "./phonemization_stuttering_test/complex_consonants_non_phonemized.wav", "text": "Strength through struggle", "category": "phonemization_disabled"},
            {"path": "./phonemization_stuttering_test/technical_terms_phonemized.wav", "text": "Neural network architecture optimization", "category": "phonemization_enabled"},
            {"path": "./phonemization_stuttering_test/technical_terms_non_phonemized.wav", "text": "Neural network architecture optimization", "category": "phonemization_disabled"},
            {"path": "./phonemization_stuttering_test/stuttering_test_phonemized.wav", "text": "Testing text-to-speech stuttering artifacts", "category": "phonemization_enabled"},
            {"path": "./phonemization_stuttering_test/stuttering_test_non_phonemized.wav", "text": "Testing text-to-speech stuttering artifacts", "category": "phonemization_disabled"},
            
            # Enhancement Pipeline Test Samples
            {"path": "./stuttering_investigation/simple_word_enhancement_enabled.wav", "text": "Welcome", "category": "enhancement_enabled"},
            {"path": "./stuttering_investigation/simple_word_enhancement_disabled.wav", "text": "Welcome", "category": "enhancement_disabled"},
            {"path": "./stuttering_investigation/simple_word_raw_model_output.wav", "text": "Welcome", "category": "raw_model_output"},
            {"path": "./stuttering_investigation/stuttering_trigger_enhancement_enabled.wav", "text": "Text-to-speech synthesis", "category": "enhancement_enabled"},
            {"path": "./stuttering_investigation/stuttering_trigger_enhancement_disabled.wav", "text": "Text-to-speech synthesis", "category": "enhancement_disabled"},
            {"path": "./stuttering_investigation/stuttering_trigger_raw_model_output.wav", "text": "Text-to-speech synthesis", "category": "raw_model_output"},
            {"path": "./stuttering_investigation/complex_phrase_enhancement_enabled.wav", "text": "The quick brown fox jumps over the lazy dog", "category": "enhancement_enabled"},
            {"path": "./stuttering_investigation/complex_phrase_enhancement_disabled.wav", "text": "The quick brown fox jumps over the lazy dog", "category": "enhancement_disabled"},
            {"path": "./stuttering_investigation/complex_phrase_raw_model_output.wav", "text": "The quick brown fox jumps over the lazy dog", "category": "raw_model_output"},
            {"path": "./stuttering_investigation/technical_terms_enhancement_enabled.wav", "text": "Neural network architecture optimization", "category": "enhancement_enabled"},
            {"path": "./stuttering_investigation/technical_terms_enhancement_disabled.wav", "text": "Neural network architecture optimization", "category": "enhancement_disabled"},
            {"path": "./stuttering_investigation/technical_terms_raw_model_output.wav", "text": "Neural network architecture optimization", "category": "raw_model_output"},
            
            # Torch Compile Test Samples
            {"path": "./torch_compile_stuttering_test/welcome_word_compiled.wav", "text": "Welcome", "category": "torch_compiled"},
            {"path": "./torch_compile_stuttering_test/welcome_word_non_compiled.wav", "text": "Welcome", "category": "torch_non_compiled"},
            {"path": "./torch_compile_stuttering_test/tts_phrase_compiled.wav", "text": "Text-to-speech synthesis", "category": "torch_compiled"},
            {"path": "./torch_compile_stuttering_test/tts_phrase_non_compiled.wav", "text": "Text-to-speech synthesis", "category": "torch_non_compiled"},
            {"path": "./torch_compile_stuttering_test/stuttering_test_compiled.wav", "text": "Testing text-to-speech stuttering artifacts", "category": "torch_compiled"},
            {"path": "./torch_compile_stuttering_test/stuttering_test_non_compiled.wav", "text": "Testing text-to-speech stuttering artifacts", "category": "torch_non_compiled"},
        ]
        
        self.assessment_data["total_samples"] = len(self.audio_samples)
        
        print(f"Manual Audio Quality Assessment Tool")
        print(f"Total samples to assess: {len(self.audio_samples)}")
        print(f"Assessment categories: {set(sample['category'] for sample in self.audio_samples)}")
        print()
    
    def run_interactive_assessment(self) -> Dict[str, Any]:
        """Run interactive manual audio assessment."""
        print("=== MANUAL AUDIO QUALITY ASSESSMENT ===\n")
        print("This tool will guide you through systematic listening analysis of all generated audio samples.")
        print("For each sample, you'll provide ratings and identify specific quality issues.\n")
        
        print("ASSESSMENT CRITERIA:")
        print("• Intelligibility Score: 0-10 (0=unintelligible, 10=perfectly clear)")
        print("• Naturalness Rating: 1-5 (1=robotic, 5=human-like)")
        print("• Specific Artifacts: T-T-S fragmentation, syllable separation, distortion, clipping")
        print("• Audio Characteristics: Pitch stability, rhythm, prosody, background noise\n")
        
        # Check if we can play audio
        audio_player = self._detect_audio_player()
        if not audio_player:
            print("⚠️ No audio player detected. You'll need to manually play audio files.")
            print("   Suggested players: aplay, paplay, mpv, vlc\n")
        
        proceed = input("Ready to start assessment? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Assessment cancelled.")
            return self.assessment_data
        
        # Assess each sample
        for i, sample in enumerate(self.audio_samples, 1):
            print(f"\n{'='*60}")
            print(f"SAMPLE {i}/{len(self.audio_samples)}")
            print(f"{'='*60}")
            
            assessment = self._assess_single_sample(sample, audio_player)
            self.assessment_data["samples"][sample["path"]] = assessment
            
            # Save progress after each sample
            self._save_progress()
        
        # Generate summary analysis
        self._generate_summary_analysis()
        
        # Save final results
        self._save_final_results()
        
        print(f"\n✅ Manual audio assessment completed!")
        print(f"📊 Results saved to: MANUAL_AUDIO_ASSESSMENT.md")
        print(f"📁 Raw data saved to: manual_audio_assessment_data.json")
        
        return self.assessment_data
    
    def _detect_audio_player(self) -> Optional[str]:
        """Detect available audio player."""
        players = ['aplay', 'paplay', 'mpv', 'vlc', 'ffplay']
        
        for player in players:
            try:
                result = subprocess.run(['which', player], capture_output=True, text=True)
                if result.returncode == 0:
                    return player
            except:
                continue
        
        return None
    
    def _assess_single_sample(self, sample: Dict[str, Any], audio_player: Optional[str]) -> Dict[str, Any]:
        """Assess a single audio sample."""
        print(f"File: {sample['path']}")
        print(f"Text: \"{sample['text']}\"")
        print(f"Category: {sample['category']}")
        print()
        
        # Check if file exists
        if not Path(sample["path"]).exists():
            print(f"❌ File not found: {sample['path']}")
            return {"error": "File not found", "skipped": True}
        
        # Play audio if player available
        if audio_player:
            play_again = True
            while play_again:
                try:
                    print(f"🔊 Playing audio with {audio_player}...")
                    subprocess.run([audio_player, sample["path"]], check=True, capture_output=True)
                    print("✅ Playback completed")
                except Exception as e:
                    print(f"❌ Playback failed: {e}")
                
                replay = input("Play again? (y/n): ").lower().strip()
                play_again = replay == 'y'
        else:
            print(f"🎧 Please manually play: {sample['path']}")
            input("Press Enter when you've listened to the audio...")
        
        # Collect assessment data
        assessment = {
            "file_path": sample["path"],
            "text": sample["text"],
            "category": sample["category"],
            "assessment_timestamp": time.time()
        }
        
        # Intelligibility Score (0-10)
        while True:
            try:
                score = input("Intelligibility Score (0-10, 0=unintelligible, 10=perfectly clear): ").strip()
                intelligibility = int(score)
                if 0 <= intelligibility <= 10:
                    assessment["intelligibility_score"] = intelligibility
                    break
                else:
                    print("Please enter a number between 0 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Naturalness Rating (1-5)
        while True:
            try:
                rating = input("Naturalness Rating (1-5, 1=robotic, 5=human-like): ").strip()
                naturalness = int(rating)
                if 1 <= naturalness <= 5:
                    assessment["naturalness_rating"] = naturalness
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        # Specific Artifacts
        print("\nSpecific Artifacts (check all that apply):")
        print("1. T-T-S fragmentation (syllable separation)")
        print("2. Distortion/clipping")
        print("3. Unnatural pitch/prosody")
        print("4. Background noise/artifacts")
        print("5. Robotic/mechanical sound")
        print("6. Other")
        
        artifacts = input("Enter numbers separated by commas (e.g., 1,3,5) or 'none': ").strip()
        
        artifact_list = []
        if artifacts.lower() != 'none':
            try:
                artifact_numbers = [int(x.strip()) for x in artifacts.split(',') if x.strip()]
                artifact_map = {
                    1: "T-T-S fragmentation",
                    2: "Distortion/clipping", 
                    3: "Unnatural pitch/prosody",
                    4: "Background noise/artifacts",
                    5: "Robotic/mechanical sound",
                    6: "Other"
                }
                artifact_list = [artifact_map.get(num, f"Unknown({num})") for num in artifact_numbers if num in artifact_map]
            except ValueError:
                artifact_list = ["Parse error"]
        
        assessment["artifacts"] = artifact_list
        
        # Additional notes
        notes = input("Additional notes (optional): ").strip()
        if notes:
            assessment["notes"] = notes
        
        # Overall quality assessment
        if intelligibility >= 8 and naturalness >= 4:
            quality_level = "Excellent"
        elif intelligibility >= 6 and naturalness >= 3:
            quality_level = "Good"
        elif intelligibility >= 4 and naturalness >= 2:
            quality_level = "Fair"
        elif intelligibility >= 2:
            quality_level = "Poor"
        else:
            quality_level = "Unacceptable"
        
        assessment["overall_quality"] = quality_level
        
        print(f"✅ Assessment completed - Overall Quality: {quality_level}")
        
        return assessment

    def _save_progress(self) -> None:
        """Save assessment progress."""
        with open("manual_audio_assessment_progress.json", "w") as f:
            json.dump(self.assessment_data, f, indent=2, default=str)

    def _generate_summary_analysis(self) -> None:
        """Generate summary analysis of all assessments."""
        samples = self.assessment_data["samples"]
        valid_samples = {k: v for k, v in samples.items() if not v.get("skipped", False)}

        if not valid_samples:
            return

        # Calculate statistics
        intelligibility_scores = [s["intelligibility_score"] for s in valid_samples.values()]
        naturalness_ratings = [s["naturalness_rating"] for s in valid_samples.values()]

        self.assessment_data["summary_statistics"] = {
            "total_assessed": len(valid_samples),
            "average_intelligibility": sum(intelligibility_scores) / len(intelligibility_scores),
            "average_naturalness": sum(naturalness_ratings) / len(naturalness_ratings),
            "intelligibility_distribution": {
                "excellent (8-10)": len([s for s in intelligibility_scores if s >= 8]),
                "good (6-7)": len([s for s in intelligibility_scores if 6 <= s < 8]),
                "fair (4-5)": len([s for s in intelligibility_scores if 4 <= s < 6]),
                "poor (2-3)": len([s for s in intelligibility_scores if 2 <= s < 4]),
                "unacceptable (0-1)": len([s for s in intelligibility_scores if s < 2])
            },
            "naturalness_distribution": {
                "human-like (5)": len([r for r in naturalness_ratings if r == 5]),
                "natural (4)": len([r for r in naturalness_ratings if r == 4]),
                "acceptable (3)": len([r for r in naturalness_ratings if r == 3]),
                "robotic (2)": len([r for r in naturalness_ratings if r == 2]),
                "very robotic (1)": len([r for r in naturalness_ratings if r == 1])
            }
        }

        # Quality rankings
        quality_rankings = {}
        for category in set(s["category"] for s in valid_samples.values()):
            category_samples = {k: v for k, v in valid_samples.items() if v["category"] == category}
            if category_samples:
                avg_intelligibility = sum(s["intelligibility_score"] for s in category_samples.values()) / len(category_samples)
                avg_naturalness = sum(s["naturalness_rating"] for s in category_samples.values()) / len(category_samples)
                quality_rankings[category] = {
                    "average_intelligibility": avg_intelligibility,
                    "average_naturalness": avg_naturalness,
                    "sample_count": len(category_samples),
                    "combined_score": (avg_intelligibility / 10 * 0.7) + (avg_naturalness / 5 * 0.3)  # Weighted score
                }

        # Sort categories by quality
        sorted_categories = sorted(quality_rankings.items(), key=lambda x: x[1]["combined_score"], reverse=True)
        self.assessment_data["quality_rankings"] = dict(sorted_categories)

        # Generate findings
        findings = []

        # Overall quality assessment
        avg_intelligibility = self.assessment_data["summary_statistics"]["average_intelligibility"]
        avg_naturalness = self.assessment_data["summary_statistics"]["average_naturalness"]

        if avg_intelligibility < 4:
            findings.append(f"CRITICAL: Average intelligibility is very low ({avg_intelligibility:.1f}/10) - most samples are unintelligible")
        elif avg_intelligibility < 6:
            findings.append(f"WARNING: Average intelligibility is below acceptable ({avg_intelligibility:.1f}/10)")
        elif avg_intelligibility >= 8:
            findings.append(f"EXCELLENT: Average intelligibility is high ({avg_intelligibility:.1f}/10)")

        if avg_naturalness < 2:
            findings.append(f"CRITICAL: Average naturalness is very low ({avg_naturalness:.1f}/5) - speech sounds very robotic")
        elif avg_naturalness < 3:
            findings.append(f"WARNING: Average naturalness is below acceptable ({avg_naturalness:.1f}/5)")
        elif avg_naturalness >= 4:
            findings.append(f"EXCELLENT: Average naturalness is high ({avg_naturalness:.1f}/5)")

        # Category-specific findings
        if quality_rankings:
            best_category = sorted_categories[0]
            worst_category = sorted_categories[-1]

            findings.append(f"BEST PERFORMING: {best_category[0]} (combined score: {best_category[1]['combined_score']:.3f})")
            findings.append(f"WORST PERFORMING: {worst_category[0]} (combined score: {worst_category[1]['combined_score']:.3f})")

        # Artifact analysis
        all_artifacts = []
        for sample in valid_samples.values():
            all_artifacts.extend(sample.get("artifacts", []))

        if all_artifacts:
            from collections import Counter
            artifact_counts = Counter(all_artifacts)
            most_common = artifact_counts.most_common(3)
            findings.append(f"MOST COMMON ARTIFACTS: {', '.join([f'{artifact} ({count} samples)' for artifact, count in most_common])}")

        self.assessment_data["findings"] = findings

    def _save_final_results(self) -> None:
        """Save final assessment results."""
        # Save raw data
        with open("manual_audio_assessment_data.json", "w") as f:
            json.dump(self.assessment_data, f, indent=2, default=str)

        # Generate markdown report
        self._generate_markdown_report()

    def _generate_markdown_report(self) -> None:
        """Generate comprehensive markdown assessment report."""
        with open("MANUAL_AUDIO_ASSESSMENT.md", "w") as f:
            f.write("# Manual Audio Quality Assessment Report\n\n")
            f.write(f"**Assessment Date**: {self.assessment_data['assessment_date']}\n")
            f.write(f"**Total Samples Assessed**: {self.assessment_data['summary_statistics'].get('total_assessed', 0)}\n")
            f.write(f"**Assessment Duration**: Comprehensive human listening analysis\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            stats = self.assessment_data["summary_statistics"]
            f.write(f"- **Average Intelligibility**: {stats.get('average_intelligibility', 0):.1f}/10\n")
            f.write(f"- **Average Naturalness**: {stats.get('average_naturalness', 0):.1f}/5\n")
            f.write(f"- **Overall Assessment**: ")

            avg_intelligibility = stats.get('average_intelligibility', 0)
            avg_naturalness = stats.get('average_naturalness', 0)

            if avg_intelligibility >= 8 and avg_naturalness >= 4:
                f.write("EXCELLENT - High quality, natural speech\n")
            elif avg_intelligibility >= 6 and avg_naturalness >= 3:
                f.write("GOOD - Acceptable quality with minor issues\n")
            elif avg_intelligibility >= 4 and avg_naturalness >= 2:
                f.write("FAIR - Noticeable quality issues affecting usability\n")
            elif avg_intelligibility >= 2:
                f.write("POOR - Significant quality degradation\n")
            else:
                f.write("UNACCEPTABLE - Severe quality issues, mostly unintelligible\n")
            f.write("\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(self.assessment_data.get("findings", []), 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")

            # Quality Rankings by Category
            f.write("## Quality Rankings by Category\n\n")
            f.write("| Rank | Category | Avg Intelligibility | Avg Naturalness | Combined Score | Sample Count |\n")
            f.write("|------|----------|-------------------|------------------|----------------|-------------|\n")

            for i, (category, data) in enumerate(self.assessment_data.get("quality_rankings", {}).items(), 1):
                f.write(f"| {i} | {category} | {data['average_intelligibility']:.1f}/10 | ")
                f.write(f"{data['average_naturalness']:.1f}/5 | {data['combined_score']:.3f} | {data['sample_count']} |\n")
            f.write("\n")

            # Detailed Sample Results
            f.write("## Detailed Sample Assessment Results\n\n")

            # Group by category
            samples_by_category = {}
            for path, sample in self.assessment_data["samples"].items():
                if not sample.get("skipped", False):
                    category = sample["category"]
                    if category not in samples_by_category:
                        samples_by_category[category] = []
                    samples_by_category[category].append((path, sample))

            for category, samples in samples_by_category.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                f.write("| Sample | Text | Intelligibility | Naturalness | Quality | Artifacts |\n")
                f.write("|--------|------|----------------|-------------|---------|----------|\n")

                for path, sample in samples:
                    filename = Path(path).name
                    text = sample["text"][:30] + "..." if len(sample["text"]) > 30 else sample["text"]
                    artifacts = ", ".join(sample.get("artifacts", ["None"]))

                    f.write(f"| {filename} | {text} | {sample['intelligibility_score']}/10 | ")
                    f.write(f"{sample['naturalness_rating']}/5 | {sample['overall_quality']} | {artifacts} |\n")
                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            avg_intelligibility = stats.get('average_intelligibility', 0)
            if avg_intelligibility < 6:
                f.write("### IMMEDIATE ACTIONS REQUIRED\n")
                f.write("1. **CRITICAL**: Address fundamental audio quality issues - current intelligibility is unacceptable\n")
                f.write("2. **INVESTIGATE**: Focus on worst-performing categories for targeted improvements\n")
                f.write("3. **VALIDATE**: Implement audio format and encoding investigation (Phase 2B Task 3)\n\n")

            f.write("### NEXT STEPS\n")
            f.write("1. **Audio Format Investigation**: Proceed with systematic technical analysis\n")
            f.write("2. **Processing Pipeline Review**: Focus on stages showing worst quality degradation\n")
            f.write("3. **Targeted Remediation**: Apply specific fixes based on identified root causes\n")
            f.write("4. **Re-assessment**: Conduct follow-up manual assessment after improvements\n\n")

            f.write("---\n")
            f.write("**Note**: This manual assessment provides human-perceptible quality evaluation that ")
            f.write("complements automated technical analysis. Results guide targeted remediation efforts.\n")


def main():
    """Run manual audio assessment."""
    try:
        assessment_tool = ManualAudioAssessment()
        results = assessment_tool.run_interactive_assessment()

        print("\n" + "="*60)
        print("MANUAL AUDIO ASSESSMENT COMPLETED")
        print("="*60)

        # Print summary
        stats = results.get("summary_statistics", {})
        if stats:
            print(f"\n📊 ASSESSMENT SUMMARY:")
            print(f"  • Total samples assessed: {stats.get('total_assessed', 0)}")
            print(f"  • Average intelligibility: {stats.get('average_intelligibility', 0):.1f}/10")
            print(f"  • Average naturalness: {stats.get('average_naturalness', 0):.1f}/5")

        # Print key findings
        findings = results.get("findings", [])
        if findings:
            print(f"\n🔍 KEY FINDINGS:")
            for finding in findings[:3]:
                print(f"  • {finding}")

        print(f"\n📋 Full report available in: MANUAL_AUDIO_ASSESSMENT.md")

    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user.")
    except Exception as e:
        print(f"\nAssessment failed: {e}")


if __name__ == "__main__":
    main()
