#!/usr/bin/env python3
"""
Quick Audio Quality Test for JabberTTS

This script performs a focused test of the key improvements:
1. Speed control quality
2. Performance improvements
3. Basic audio generation
"""

import asyncio
import logging
import time
import numpy as np
import soundfile as sf
from pathlib import Path

from jabbertts.inference.engine import get_inference_engine
from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_basic_generation():
    """Test basic audio generation."""
    logger.info("🎵 Testing Basic Audio Generation")
    
    inference_engine = get_inference_engine()
    
    test_text = "Hello world, this is a test of the audio generation system."
    
    try:
        start_time = time.time()
        
        result = await inference_engine.generate_speech(
            text=test_text,
            voice="alloy",
            speed=1.0,
            response_format="wav"
        )
        
        generation_time = time.time() - start_time
        audio_data = result["audio_data"]
        sample_rate = result["sample_rate"]
        duration = len(audio_data) / sample_rate
        rtf = generation_time / duration
        
        # Save audio file
        output_file = Path("temp") / "basic_generation_test.wav"
        output_file.parent.mkdir(exist_ok=True)
        sf.write(str(output_file), audio_data, sample_rate)
        
        # Audio quality metrics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        
        logger.info(f"✅ Basic generation successful:")
        logger.info(f"   Duration: {duration:.2f}s")
        logger.info(f"   Generation time: {generation_time:.2f}s")
        logger.info(f"   RTF: {rtf:.3f}")
        logger.info(f"   RMS level: {rms:.4f}")
        logger.info(f"   Peak level: {peak:.4f}")
        logger.info(f"   Audio saved to: {output_file}")
        
        return {
            "success": True,
            "duration": duration,
            "generation_time": generation_time,
            "rtf": rtf,
            "rms": rms,
            "peak": peak,
            "audio_file": str(output_file)
        }
        
    except Exception as e:
        logger.error(f"❌ Basic generation failed: {e}")
        return {"success": False, "error": str(e)}


async def test_speed_control():
    """Test speed control quality."""
    logger.info("🎛️ Testing Speed Control Quality")
    
    inference_engine = get_inference_engine()
    test_text = "This is a speed control test."
    speeds = [0.5, 1.0, 2.0]
    results = {}
    
    for speed in speeds:
        try:
            logger.info(f"Testing speed: {speed}x")
            
            start_time = time.time()
            
            result = await inference_engine.generate_speech(
                text=test_text,
                voice="alloy",
                speed=speed,
                response_format="wav"
            )
            
            generation_time = time.time() - start_time
            audio_data = result["audio_data"]
            sample_rate = result["sample_rate"]
            duration = len(audio_data) / sample_rate
            rtf = generation_time / duration
            
            # Save audio file
            output_file = Path("temp") / f"speed_test_{speed}x_improved.wav"
            sf.write(str(output_file), audio_data, sample_rate)
            
            # Audio quality metrics
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            
            results[speed] = {
                "success": True,
                "duration": duration,
                "generation_time": generation_time,
                "rtf": rtf,
                "rms": rms,
                "peak": peak,
                "audio_file": str(output_file)
            }
            
            logger.info(f"   Speed {speed}x: Duration={duration:.2f}s, RTF={rtf:.3f}, RMS={rms:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Speed test failed for {speed}x: {e}")
            results[speed] = {"success": False, "error": str(e)}
    
    return results


async def test_voice_consistency():
    """Test voice consistency."""
    logger.info("🎭 Testing Voice Consistency")
    
    inference_engine = get_inference_engine()
    test_text = "Hello, this is a voice consistency test."
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    results = {}
    
    for voice in voices:
        try:
            logger.info(f"Testing voice: {voice}")
            
            start_time = time.time()
            
            result = await inference_engine.generate_speech(
                text=test_text,
                voice=voice,
                speed=1.0,
                response_format="wav"
            )
            
            generation_time = time.time() - start_time
            audio_data = result["audio_data"]
            sample_rate = result["sample_rate"]
            duration = len(audio_data) / sample_rate
            rtf = generation_time / duration
            
            # Save audio file
            output_file = Path("temp") / f"voice_test_{voice}_improved.wav"
            sf.write(str(output_file), audio_data, sample_rate)
            
            # Audio quality metrics
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            
            results[voice] = {
                "success": True,
                "duration": duration,
                "generation_time": generation_time,
                "rtf": rtf,
                "rms": rms,
                "peak": peak,
                "audio_file": str(output_file)
            }
            
            logger.info(f"   Voice {voice}: Duration={duration:.2f}s, RTF={rtf:.3f}, RMS={rms:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Voice test failed for {voice}: {e}")
            results[voice] = {"success": False, "error": str(e)}
    
    return results


async def test_whisper_validation():
    """Test Whisper validation with a simple case."""
    logger.info("🎤 Testing Whisper Validation")
    
    try:
        inference_engine = get_inference_engine()
        whisper_validator = get_whisper_validator("base")
        
        test_text = "Hello world"
        
        # Generate audio
        result = await inference_engine.generate_speech(
            text=test_text,
            voice="alloy",
            speed=1.0,
            response_format="wav"
        )
        
        # Convert to bytes for Whisper
        import io
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, result["audio_data"], result["sample_rate"], format='WAV')
        audio_bytes.seek(0)
        
        # Validate with Whisper
        validation_result = whisper_validator.validate_tts_output(
            original_text=test_text,
            audio_data=audio_bytes.getvalue(),
            sample_rate=result["sample_rate"]
        )
        
        accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
        transcription = validation_result.get("transcription", "")
        
        logger.info(f"✅ Whisper validation successful:")
        logger.info(f"   Original: '{test_text}'")
        logger.info(f"   Transcribed: '{transcription}'")
        logger.info(f"   Accuracy: {accuracy:.1f}%")
        
        return {
            "success": True,
            "original_text": test_text,
            "transcription": transcription,
            "accuracy": accuracy
        }
        
    except Exception as e:
        logger.error(f"❌ Whisper validation failed: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main test execution."""
    logger.info("🚀 Starting Quick Audio Quality Test")
    logger.info("=" * 50)
    
    results = {}
    
    # Test 1: Basic Generation
    results["basic_generation"] = await test_basic_generation()
    
    # Test 2: Speed Control
    results["speed_control"] = await test_speed_control()
    
    # Test 3: Voice Consistency
    results["voice_consistency"] = await test_voice_consistency()
    
    # Test 4: Whisper Validation
    results["whisper_validation"] = await test_whisper_validation()
    
    # Summary
    logger.info("\n📊 TEST SUMMARY")
    logger.info("=" * 30)
    
    # Basic generation summary
    basic = results.get("basic_generation", {})
    if basic.get("success"):
        logger.info(f"✅ Basic Generation: RTF={basic.get('rtf', 0):.3f}")
    else:
        logger.info(f"❌ Basic Generation: Failed")
    
    # Speed control summary
    speed_results = results.get("speed_control", {})
    successful_speeds = [speed for speed, data in speed_results.items() if data.get("success")]
    logger.info(f"✅ Speed Control: {len(successful_speeds)}/3 speeds successful")
    
    # Voice consistency summary
    voice_results = results.get("voice_consistency", {})
    successful_voices = [voice for voice, data in voice_results.items() if data.get("success")]
    logger.info(f"✅ Voice Consistency: {len(successful_voices)}/6 voices successful")
    
    # Whisper validation summary
    whisper = results.get("whisper_validation", {})
    if whisper.get("success"):
        logger.info(f"✅ Whisper Validation: {whisper.get('accuracy', 0):.1f}% accuracy")
    else:
        logger.info(f"❌ Whisper Validation: Failed")
    
    # Performance assessment
    if basic.get("success"):
        rtf = basic.get("rtf", float('inf'))
        if rtf < 1.0:
            logger.info(f"🎯 Performance: GOOD (RTF < 1.0)")
        elif rtf < 2.0:
            logger.info(f"⚠️ Performance: ACCEPTABLE (RTF < 2.0)")
        else:
            logger.info(f"❌ Performance: NEEDS IMPROVEMENT (RTF > 2.0)")
    
    logger.info("\n🎉 Quick Audio Quality Test Complete!")
    
    # Save results
    import json
    output_file = Path("temp") / "quick_audio_quality_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📁 Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
