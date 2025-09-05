#!/usr/bin/env python3
"""
SpeechT5 Preprocessing Fix

This script implements the critical fix for the SpeechT5 intelligibility issue.
The problem is that we're feeding phonemes to SpeechT5, but it expects raw text.

Usage:
    python jabbertts/scripts/fix_speecht5_preprocessing.py
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_preprocessing_fix():
    """Test the preprocessing fix for SpeechT5."""
    logger.info("🔧 Testing SpeechT5 Preprocessing Fix")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        whisper_validator = get_whisper_validator("base")
        
        test_text = "Hello world, this is a test."
        test_voice = "alloy"
        
        logger.info(f"Test Text: '{test_text}'")
        logger.info(f"Test Voice: {test_voice}")
        logger.info("")
        
        # Test 1: Current implementation (with phonemization)
        logger.info("Test 1: Current implementation (with phonemization)")
        logger.info("-" * 40)
        
        result_with_phonemes = await inference_engine.generate_speech(
            text=test_text,
            voice=test_voice,
            response_format="wav"
        )
        
        # Process and transcribe
        audio_data, _ = await audio_processor.process_audio(
            audio_array=result_with_phonemes["audio_data"],
            sample_rate=result_with_phonemes["sample_rate"],
            output_format="wav"
        )
        
        validation_result = whisper_validator.validate_tts_output(
            original_text=test_text,
            audio_data=audio_data,
            sample_rate=result_with_phonemes["sample_rate"]
        )
        
        accuracy_with_phonemes = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
        transcription_with_phonemes = validation_result.get("transcription", "")
        
        logger.info(f"With Phonemization:")
        logger.info(f"  Transcription: '{transcription_with_phonemes}'")
        logger.info(f"  Accuracy: {accuracy_with_phonemes:.1f}%")
        logger.info(f"  RTF: {result_with_phonemes.get('rtf', 0):.3f}")
        logger.info("")
        
        # Test 2: Apply the fix (disable phonemization for SpeechT5)
        logger.info("Test 2: Fixed implementation (no phonemization)")
        logger.info("-" * 40)
        
        # Temporarily disable phonemization in the preprocessor
        preprocessor = inference_engine.preprocessor

        # Save original setting
        original_use_phonemizer = preprocessor.use_phonemizer
        
        # Disable phonemization for SpeechT5
        preprocessor.use_phonemizer = False
        logger.info("Disabled phonemization for SpeechT5")
        
        try:
            result_without_phonemes = await inference_engine.generate_speech(
                text=test_text,
                voice=test_voice,
                response_format="wav"
            )
            
            # Process and transcribe
            audio_data_fixed, _ = await audio_processor.process_audio(
                audio_array=result_without_phonemes["audio_data"],
                sample_rate=result_without_phonemes["sample_rate"],
                output_format="wav"
            )
            
            validation_result_fixed = whisper_validator.validate_tts_output(
                original_text=test_text,
                audio_data=audio_data_fixed,
                sample_rate=result_without_phonemes["sample_rate"]
            )
            
            accuracy_without_phonemes = validation_result_fixed.get("accuracy_metrics", {}).get("overall_accuracy", 0)
            transcription_without_phonemes = validation_result_fixed.get("transcription", "")
            
            logger.info(f"Without Phonemization:")
            logger.info(f"  Transcription: '{transcription_without_phonemes}'")
            logger.info(f"  Accuracy: {accuracy_without_phonemes:.1f}%")
            logger.info(f"  RTF: {result_without_phonemes.get('rtf', 0):.3f}")
            logger.info("")
            
        finally:
            # Restore original setting
            preprocessor.use_phonemizer = original_use_phonemizer
        
        # Compare results
        logger.info("🔍 COMPARISON RESULTS")
        logger.info("=" * 30)
        
        improvement = accuracy_without_phonemes - accuracy_with_phonemes
        
        logger.info(f"Original Text: '{test_text}'")
        logger.info("")
        logger.info(f"WITH Phonemization:")
        logger.info(f"  Transcription: '{transcription_with_phonemes}'")
        logger.info(f"  Accuracy: {accuracy_with_phonemes:.1f}%")
        logger.info("")
        logger.info(f"WITHOUT Phonemization:")
        logger.info(f"  Transcription: '{transcription_without_phonemes}'")
        logger.info(f"  Accuracy: {accuracy_without_phonemes:.1f}%")
        logger.info("")
        logger.info(f"Improvement: {improvement:.1f} percentage points")
        
        # Determine if fix is successful
        if improvement > 50:
            logger.info("🎉 MAJOR IMPROVEMENT DETECTED!")
            logger.info("✅ The fix significantly improves intelligibility")
        elif improvement > 10:
            logger.info("✅ IMPROVEMENT DETECTED!")
            logger.info("The fix improves intelligibility")
        elif improvement > 0:
            logger.info("✅ Minor improvement detected")
        else:
            logger.warning("⚠️ No improvement or degradation detected")
        
        # Save results
        results = {
            "preprocessing_fix_test": {
                "timestamp": datetime.now().isoformat(),
                "test_text": test_text,
                "test_voice": test_voice,
                "with_phonemization": {
                    "transcription": transcription_with_phonemes,
                    "accuracy": accuracy_with_phonemes,
                    "rtf": result_with_phonemes.get("rtf", 0)
                },
                "without_phonemization": {
                    "transcription": transcription_without_phonemes,
                    "accuracy": accuracy_without_phonemes,
                    "rtf": result_without_phonemes.get("rtf", 0)
                },
                "improvement": improvement,
                "fix_successful": improvement > 10
            }
        }
        
        output_file = Path("temp") / "preprocessing_fix_test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def implement_permanent_fix():
    """Implement the permanent fix for SpeechT5 preprocessing."""
    logger.info("🔧 Implementing Permanent Fix for SpeechT5")
    logger.info("=" * 50)
    
    try:
        # The fix involves modifying the SpeechT5 model to bypass phonemization
        # This should be done by creating a model-specific preprocessing override
        
        logger.info("Permanent fix implementation:")
        logger.info("1. Modify SpeechT5Model to disable phonemization")
        logger.info("2. Update preprocessing pipeline to be model-aware")
        logger.info("3. Ensure other models can still use phonemization if needed")
        logger.info("")
        
        # Test the fix first
        test_results = await test_preprocessing_fix()
        
        if test_results and test_results["preprocessing_fix_test"]["fix_successful"]:
            logger.info("✅ Fix validation successful - proceeding with implementation")
            
            # Implementation steps would go here
            # For now, we'll document the required changes
            
            implementation_plan = {
                "fix_implementation_plan": {
                    "timestamp": datetime.now().isoformat(),
                    "status": "validated",
                    "required_changes": [
                        {
                            "file": "jabbertts/models/speecht5.py",
                            "change": "Override preprocessing to disable phonemization",
                            "method": "Add model-specific text preprocessing"
                        },
                        {
                            "file": "jabbertts/inference/preprocessing.py", 
                            "change": "Add model-aware preprocessing",
                            "method": "Check model type before applying phonemization"
                        },
                        {
                            "file": "jabbertts/inference/engine.py",
                            "change": "Pass model info to preprocessor",
                            "method": "Include model type in preprocessing call"
                        }
                    ],
                    "validation_results": test_results["preprocessing_fix_test"]
                }
            }
            
            output_file = Path("temp") / "fix_implementation_plan.json"
            with open(output_file, "w") as f:
                json.dump(implementation_plan, f, indent=2)
            
            logger.info(f"📁 Implementation plan saved to: {output_file}")
            
            return implementation_plan
        else:
            logger.error("❌ Fix validation failed - not implementing")
            return None
            
    except Exception as e:
        logger.error(f"Implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main execution function."""
    logger.info("🚨 CRITICAL FIX: SpeechT5 Preprocessing Issue")
    logger.info("=" * 60)
    logger.info("Issue: SpeechT5 is receiving phonemes instead of raw text")
    logger.info("Solution: Disable phonemization for SpeechT5 model")
    logger.info("=" * 60)
    
    try:
        # Test the fix
        test_results = await test_preprocessing_fix()
        
        if test_results:
            # Implement permanent fix if test is successful
            implementation_results = await implement_permanent_fix()
            
            if implementation_results:
                logger.info("✅ Fix validation and implementation planning completed")
                return {
                    "test_results": test_results,
                    "implementation_plan": implementation_results
                }
            else:
                logger.error("❌ Implementation planning failed")
                return {"test_results": test_results}
        else:
            logger.error("❌ Fix testing failed")
            return None
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
