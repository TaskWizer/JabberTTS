#!/usr/bin/env python3
"""Human Listening Test for JabberTTS Audio Quality Validation.

This script generates test samples for human evaluation to validate
that the audio quality fixes have restored intelligible speech.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.config import get_settings


class HumanListeningTest:
    """Generate test samples for human listening validation."""
    
    def __init__(self):
        """Initialize the test."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        
        # Create output directory
        self.output_dir = Path("human_listening_test")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Human Listening Test for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print(f"Model: {self.settings.model_name}")
        print(f"Audio quality: {self.settings.audio_quality}")
        print()
    
    async def generate_test_samples(self):
        """Generate test samples for human evaluation."""
        print("=== Generating Human Listening Test Samples ===\n")
        
        # Test cases designed for human intelligibility validation
        test_cases = [
            {
                "name": "basic_intelligibility",
                "text": "Hello, this is a test of speech intelligibility.",
                "description": "Basic intelligibility test - should be clearly understandable"
            },
            {
                "name": "complex_sentence",
                "text": "The quick brown fox jumps over the lazy dog near the riverbank.",
                "description": "Complex sentence with varied phonemes"
            },
            {
                "name": "numbers_and_dates",
                "text": "Today is January 15th, 2024. The temperature is 72 degrees Fahrenheit.",
                "description": "Numbers and dates pronunciation test"
            },
            {
                "name": "technical_terms",
                "text": "Text-to-speech synthesis using neural networks and machine learning algorithms.",
                "description": "Technical terminology pronunciation"
            },
            {
                "name": "natural_conversation",
                "text": "How are you doing today? I hope you're having a wonderful time!",
                "description": "Natural conversational speech with emotion"
            },
            {
                "name": "difficult_words",
                "text": "Colonel, yacht, psychology, choir, and schedule are difficult to pronounce.",
                "description": "Challenging pronunciation words"
            }
        ]
        
        # Test all supported formats
        formats = ["wav", "mp3", "flac", "opus"]
        
        results = {
            "test_timestamp": time.time(),
            "test_cases": {},
            "format_comparison": {},
            "instructions": {}
        }
        
        # Generate samples for each test case
        for test_case in test_cases:
            print(f"Generating: {test_case['name']}")
            print(f"  Text: '{test_case['text']}'")
            print(f"  Purpose: {test_case['description']}")
            
            try:
                # Generate speech
                result = await self.engine.generate_speech(test_case["text"], voice="alloy")
                audio_data = result["audio_data"]
                sample_rate = result["sample_rate"]
                
                # Save in all formats for comparison
                format_files = {}
                for format_name in formats:
                    try:
                        # Process audio to format
                        encoded_audio, metadata = await self.audio_processor.process_audio(
                            audio_data, sample_rate, output_format=format_name
                        )
                        
                        # Save file
                        filename = self.output_dir / f"{test_case['name']}.{format_name}"
                        with open(filename, 'wb') as f:
                            f.write(encoded_audio)
                        
                        format_files[format_name] = {
                            "filename": str(filename),
                            "size": len(encoded_audio),
                            "metadata": metadata
                        }
                        
                        print(f"    ✓ {format_name.upper()}: {len(encoded_audio)} bytes")
                        
                    except Exception as e:
                        print(f"    ✗ {format_name.upper()} failed: {e}")
                        format_files[format_name] = {"error": str(e)}
                
                results["test_cases"][test_case["name"]] = {
                    "text": test_case["text"],
                    "description": test_case["description"],
                    "files": format_files,
                    "audio_info": {
                        "duration": result["duration"],
                        "sample_rate": sample_rate,
                        "rtf": result["rtf"]
                    }
                }
                
                print(f"    ✓ Duration: {result['duration']:.2f}s, RTF: {result['rtf']:.3f}")
                print()
                
            except Exception as e:
                print(f"    ✗ Failed to generate {test_case['name']}: {e}")
                results["test_cases"][test_case["name"]] = {"error": str(e)}
                print()
        
        # Generate format comparison with same text
        print("Generating format comparison sample...")
        comparison_text = "This sample allows direct comparison of audio quality across different formats."
        
        try:
            result = await self.engine.generate_speech(comparison_text, voice="alloy")
            audio_data = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            for format_name in formats:
                try:
                    encoded_audio, metadata = await self.audio_processor.process_audio(
                        audio_data, sample_rate, output_format=format_name
                    )
                    
                    filename = self.output_dir / f"format_comparison.{format_name}"
                    with open(filename, 'wb') as f:
                        f.write(encoded_audio)
                    
                    results["format_comparison"][format_name] = {
                        "filename": str(filename),
                        "size": len(encoded_audio),
                        "compression_ratio": len(audio_data) * 4 / len(encoded_audio),
                        "metadata": metadata
                    }
                    
                    print(f"  ✓ {format_name.upper()}: {len(encoded_audio)} bytes")
                    
                except Exception as e:
                    print(f"  ✗ {format_name.upper()} failed: {e}")
                    results["format_comparison"][format_name] = {"error": str(e)}
        
        except Exception as e:
            print(f"  ✗ Format comparison failed: {e}")
        
        # Generate instructions for human evaluation
        self._generate_evaluation_instructions(results)
        
        # Save results
        import json
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Human listening test samples generated!")
        print(f"✓ Files saved to: {self.output_dir}")
        print(f"✓ Results: {results_file}")
        print(f"\n📋 Please review the evaluation instructions in:")
        print(f"   {self.output_dir}/HUMAN_EVALUATION_INSTRUCTIONS.md")
        
        return results
    
    def _generate_evaluation_instructions(self, results):
        """Generate human evaluation instructions."""
        instructions_file = self.output_dir / "HUMAN_EVALUATION_INSTRUCTIONS.md"
        
        with open(instructions_file, 'w') as f:
            f.write("# Human Listening Test - Evaluation Instructions\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Purpose**: Validate audio quality improvements in JabberTTS\n\n")
            
            f.write("## Test Objective\n\n")
            f.write("Evaluate whether the audio quality fixes have restored human-intelligible speech.\n")
            f.write("Previous issues included stuttering, robotic artifacts, and unintelligible output.\n\n")
            
            f.write("## Evaluation Criteria\n\n")
            f.write("For each audio sample, rate the following on a scale of 1-5:\n\n")
            f.write("1. **Intelligibility** (1=unintelligible, 5=perfectly clear)\n")
            f.write("2. **Naturalness** (1=robotic, 5=human-like)\n")
            f.write("3. **Audio Quality** (1=poor/distorted, 5=excellent)\n")
            f.write("4. **Overall Acceptability** (1=unacceptable, 5=excellent)\n\n")
            
            f.write("## Test Samples\n\n")
            
            for test_name, test_data in results.get("test_cases", {}).items():
                if "error" not in test_data:
                    f.write(f"### {test_name.replace('_', ' ').title()}\n")
                    f.write(f"**Text**: \"{test_data['text']}\"\n")
                    f.write(f"**Purpose**: {test_data['description']}\n")
                    f.write("**Files**:\n")
                    
                    for format_name, file_info in test_data.get("files", {}).items():
                        if "error" not in file_info:
                            f.write(f"- {format_name.upper()}: `{Path(file_info['filename']).name}`\n")
                    f.write("\n")
            
            f.write("## Format Comparison\n\n")
            f.write("Compare the same text across different audio formats:\n")
            f.write("**Text**: \"This sample allows direct comparison of audio quality across different formats.\"\n\n")
            
            for format_name, file_info in results.get("format_comparison", {}).items():
                if "error" not in file_info:
                    f.write(f"- {format_name.upper()}: `{Path(file_info['filename']).name}` ")
                    f.write(f"({file_info['size']} bytes, {file_info.get('compression_ratio', 0):.1f}x compression)\n")
            
            f.write("\n## Success Criteria\n\n")
            f.write("The fixes are successful if:\n")
            f.write("- All samples are intelligible (score ≥3)\n")
            f.write("- No stuttering or robotic artifacts\n")
            f.write("- Natural speech rhythm and intonation\n")
            f.write("- Consistent quality across formats\n\n")
            
            f.write("## Previous Issues (Should be Fixed)\n\n")
            f.write("- ❌ Stuttering/fragmented speech (\"T-T-S\" artifacts)\n")
            f.write("- ❌ Robotic/machine-like quality\n")
            f.write("- ❌ Muffled or distorted audio (especially MP3)\n")
            f.write("- ❌ Unintelligible speech output\n")
            f.write("- ❌ Excessive silence or gaps\n\n")
            
            f.write("## Expected Results (After Fixes)\n\n")
            f.write("- ✅ Clear, intelligible speech\n")
            f.write("- ✅ Natural human-like voice quality\n")
            f.write("- ✅ Consistent quality across formats\n")
            f.write("- ✅ No audio artifacts or distortion\n")
            f.write("- ✅ Proper speech rhythm and timing\n\n")
            
            f.write("---\n")
            f.write("**Note**: Listen to samples with good quality headphones or speakers for accurate evaluation.\n")


async def main():
    """Run the human listening test."""
    test = HumanListeningTest()
    
    try:
        results = await test.generate_test_samples()
        print("\n=== Human Listening Test Complete ===")
        print("Please evaluate the generated audio samples and provide feedback.")
        return results
    except Exception as e:
        print(f"\n✗ Human listening test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
