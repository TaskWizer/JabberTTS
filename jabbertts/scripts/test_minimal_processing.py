#!/usr/bin/env python3
"""
Test minimal processing pipeline to isolate audio quality issues.

This script tests the TTS pipeline with progressively minimal processing
to identify where audio intelligibility is lost.
"""

import asyncio
import logging
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalProcessingTester:
    """Test minimal processing pipeline configurations."""
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.settings = get_settings()
        self.test_phrases = [
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing audio quality with minimal processing.",
        ]
        
    async def test_raw_model_output(self, text: str, voice: str = "alloy") -> dict:
        """Test raw model output without any post-processing."""
        logger.info(f"Testing raw model output for: '{text}'")
        
        try:
            # Generate TTS with minimal settings
            result = await self.inference_engine.generate_speech(
                text=text,
                voice=voice,
                speed=1.0,
                response_format="wav"
            )
            
            # Save raw output directly
            raw_audio = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            # Save as raw WAV without any processing
            output_path = f"test_raw_{voice}_{datetime.now().strftime('%H%M%S')}.wav"
            sf.write(output_path, raw_audio, sample_rate)
            
            return {
                "stage": "raw_model",
                "audio_path": output_path,
                "duration": len(raw_audio) / sample_rate,
                "sample_rate": sample_rate,
                "audio_shape": raw_audio.shape,
                "audio_dtype": str(raw_audio.dtype),
                "audio_range": (float(raw_audio.min()), float(raw_audio.max())),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Raw model test failed: {e}")
            return {"stage": "raw_model", "success": False, "error": str(e)}
    
    async def test_minimal_processing(self, text: str, voice: str = "alloy") -> dict:
        """Test with minimal audio processing (normalization only)."""
        logger.info(f"Testing minimal processing for: '{text}'")
        
        try:
            # Generate TTS
            result = await self.inference_engine.generate_speech(
                text=text,
                voice=voice,
                speed=1.0,
                response_format="wav"
            )
            
            # Apply only basic normalization
            audio_array = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            # Manual minimal processing
            # 1. Ensure float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # 2. Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array[:, 0] if audio_array.shape[1] > 0 else audio_array.flatten()
            
            # 3. Basic normalization only
            max_val = np.abs(audio_array).max()
            if max_val > 0.85:
                audio_array = audio_array / max_val * 0.85
            
            # Save minimal processed audio
            output_path = f"test_minimal_{voice}_{datetime.now().strftime('%H%M%S')}.wav"
            sf.write(output_path, audio_array, sample_rate)
            
            return {
                "stage": "minimal_processing",
                "audio_path": output_path,
                "duration": len(audio_array) / sample_rate,
                "sample_rate": sample_rate,
                "audio_shape": audio_array.shape,
                "audio_dtype": str(audio_array.dtype),
                "audio_range": (float(audio_array.min()), float(audio_array.max())),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Minimal processing test failed: {e}")
            return {"stage": "minimal_processing", "success": False, "error": str(e)}
    
    async def test_no_enhancement_processing(self, text: str, voice: str = "alloy") -> dict:
        """Test with AudioProcessor but enhancement disabled."""
        logger.info(f"Testing no enhancement processing for: '{text}'")
        
        try:
            # Generate TTS
            result = await self.inference_engine.generate_speech(
                text=text,
                voice=voice,
                speed=1.0,
                response_format="wav"
            )
            
            # Temporarily disable enhancement
            original_enhancement = self.settings.enable_audio_enhancement
            self.settings.enable_audio_enhancement = False
            
            # Process with enhancement disabled
            processed_audio, metadata = await self.audio_processor.process_audio(
                audio_array=result["audio_data"],
                sample_rate=result["sample_rate"],
                output_format="wav",
                speed=1.0
            )
            
            # Restore original setting
            self.settings.enable_audio_enhancement = original_enhancement
            
            # Save processed audio
            output_path = f"test_no_enhancement_{voice}_{datetime.now().strftime('%H%M%S')}.wav"
            with open(output_path, "wb") as f:
                f.write(processed_audio)
            
            return {
                "stage": "no_enhancement",
                "audio_path": output_path,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"No enhancement test failed: {e}")
            return {"stage": "no_enhancement", "success": False, "error": str(e)}
    
    async def test_standard_processing(self, text: str, voice: str = "alloy") -> dict:
        """Test with standard AudioProcessor settings."""
        logger.info(f"Testing standard processing for: '{text}'")
        
        try:
            # Generate TTS
            result = await self.inference_engine.generate_speech(
                text=text,
                voice=voice,
                speed=1.0,
                response_format="wav"
            )
            
            # Process with standard settings
            processed_audio, metadata = await self.audio_processor.process_audio(
                audio_array=result["audio_data"],
                sample_rate=result["sample_rate"],
                output_format="wav",
                speed=1.0
            )
            
            # Save processed audio
            output_path = f"test_standard_{voice}_{datetime.now().strftime('%H%M%S')}.wav"
            with open(output_path, "wb") as f:
                f.write(processed_audio)
            
            return {
                "stage": "standard_processing",
                "audio_path": output_path,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Standard processing test failed: {e}")
            return {"stage": "standard_processing", "success": False, "error": str(e)}
    
    async def run_comprehensive_test(self):
        """Run comprehensive minimal processing test."""
        logger.info("Starting comprehensive minimal processing test...")
        
        test_text = "Hello world, this is a comprehensive test of minimal processing."
        test_voice = "alloy"
        
        results = []
        
        # Test 1: Raw model output
        raw_result = await self.test_raw_model_output(test_text, test_voice)
        results.append(raw_result)
        
        # Test 2: Minimal processing
        minimal_result = await self.test_minimal_processing(test_text, test_voice)
        results.append(minimal_result)
        
        # Test 3: No enhancement processing
        no_enhancement_result = await self.test_no_enhancement_processing(test_text, test_voice)
        results.append(no_enhancement_result)
        
        # Test 4: Standard processing
        standard_result = await self.test_standard_processing(test_text, test_voice)
        results.append(standard_result)
        
        # Print results
        print("\n" + "="*60)
        print("MINIMAL PROCESSING TEST RESULTS")
        print("="*60)
        
        for result in results:
            if result["success"]:
                print(f"\n{result['stage'].upper()}:")
                print(f"  Audio file: {result.get('audio_path', 'N/A')}")
                if 'duration' in result:
                    print(f"  Duration: {result['duration']:.3f}s")
                if 'sample_rate' in result:
                    print(f"  Sample rate: {result['sample_rate']}Hz")
                if 'audio_range' in result:
                    print(f"  Audio range: {result['audio_range']}")
                if 'metadata' in result:
                    print(f"  Metadata: {result['metadata']}")
            else:
                print(f"\n{result['stage'].upper()}: FAILED")
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*60)
        print("Test complete. Compare audio files to identify quality degradation point.")
        print("="*60)
        
        return results

async def main():
    """Main test function."""
    tester = MinimalProcessingTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
