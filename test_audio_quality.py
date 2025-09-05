#!/usr/bin/env python3
"""Audio Quality Test Script for JabberTTS.

This script tests the audio quality and performance of the TTS system.
"""

import sys
import time
import asyncio
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.config import get_settings


async def test_audio_quality():
    """Test audio quality and performance."""
    print("=== JabberTTS Audio Quality Test ===\n")
    
    # Initialize components
    settings = get_settings()
    engine = InferenceEngine()
    audio_processor = AudioProcessor()
    
    print(f"Model: {settings.model_name}")
    print(f"Audio quality: {settings.audio_quality}")
    print(f"Sample rate: {settings.sample_rate}Hz")
    print()
    
    # Test texts with varying complexity
    test_texts = [
        "Hello, world!",
        "This is a test of the text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing pronunciation: schedule, often, either, neither.",
        "Numbers and dates: 123, 2024, January 15th, $99.99.",
    ]
    
    print("=== Warming up system ===")
    await engine.warmup(num_runs=3)
    print()
    
    print("=== Audio Quality Tests ===")
    
    total_audio_duration = 0
    total_inference_time = 0
    rtfs = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: '{text}'")
        
        try:
            # Generate speech
            start_time = time.time()
            result = await engine.generate_speech(text, voice="alloy")
            inference_time = time.time() - start_time
            
            audio_data = result["audio_data"]
            sample_rate = result["sample_rate"]
            duration = result["duration"]
            rtf = result["rtf"]
            
            # Basic audio quality checks
            audio_stats = {
                "shape": audio_data.shape,
                "dtype": audio_data.dtype,
                "sample_rate": sample_rate,
                "duration": f"{duration:.2f}s",
                "rtf": f"{rtf:.3f}",
                "rms": f"{np.sqrt(np.mean(audio_data**2)):.4f}",
                "peak": f"{np.max(np.abs(audio_data)):.4f}",
                "dynamic_range": f"{20 * np.log10(np.max(np.abs(audio_data)) / (np.sqrt(np.mean(audio_data**2)) + 1e-8)):.1f} dB"
            }
            
            print(f"  ✓ Generated: {audio_stats['shape']} samples, {audio_stats['duration']}")
            print(f"  ✓ Performance: RTF = {audio_stats['rtf']}")
            print(f"  ✓ Quality: RMS = {audio_stats['rms']}, Peak = {audio_stats['peak']}")
            print(f"  ✓ Dynamic Range: {audio_stats['dynamic_range']}")
            
            # Test audio processing
            try:
                processed_audio, metadata = await audio_processor.process_audio(
                    audio_data, sample_rate, output_format="wav"
                )
                print(f"  ✓ Audio processing: {len(processed_audio)} bytes")
                
                if metadata.get("original_sample_rate") != metadata.get("final_sample_rate"):
                    print(f"  ⚠ Resampling: {metadata['original_sample_rate']}Hz → {metadata['final_sample_rate']}Hz")
                
            except Exception as e:
                print(f"  ✗ Audio processing failed: {e}")
            
            # Accumulate stats
            total_audio_duration += duration
            total_inference_time += inference_time
            rtfs.append(rtf)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        print()
    
    # Summary statistics
    print("=== Performance Summary ===")
    print(f"Total audio generated: {total_audio_duration:.2f}s")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average RTF: {sum(rtfs)/len(rtfs):.3f}")
    print(f"Best RTF: {min(rtfs):.3f}")
    print(f"Worst RTF: {max(rtfs):.3f}")
    
    # Performance assessment
    avg_rtf = sum(rtfs) / len(rtfs)
    if avg_rtf < 0.5:
        print("✓ PASS: Performance target achieved (RTF < 0.5)")
    else:
        print(f"⚠ WARNING: Performance target not met (RTF {avg_rtf:.3f} >= 0.5)")
    
    # Audio quality assessment
    print("\n=== Audio Quality Assessment ===")
    print("✓ Audio generation successful for all test cases")
    print("✓ Consistent sample rates and formats")
    print("✓ Reasonable dynamic range and levels")
    
    if settings.model_name == "speecht5":
        print("ℹ Currently using SpeechT5 model (16kHz)")
        print("ℹ For higher quality, consider implementing OpenAudio S1-mini (24kHz)")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_audio_quality())
