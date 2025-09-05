#!/usr/bin/env python3
"""
Debug Phonemization Process

This script investigates what happens during phonemization and tests
different text inputs to understand the SpeechT5 issue better.

Usage:
    python jabbertts/scripts/debug_phonemization.py
"""

import asyncio
import logging
from pathlib import Path

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def debug_phonemization():
    """Debug the phonemization process."""
    logger.info("🔍 Debugging Phonemization Process")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        whisper_validator = get_whisper_validator("base")
        
        # Test different texts
        test_cases = [
            "Hello",
            "Hello world",
            "Hello world.",
            "The cat sat on the mat.",
            "Testing one two three."
        ]
        
        for i, test_text in enumerate(test_cases, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"TEST CASE {i}: '{test_text}'")
            logger.info(f"{'='*60}")
            
            # Get the preprocessor
            preprocessor = inference_engine.preprocessor
            
            # Test phonemization directly
            logger.info("Step 1: Direct phonemization test")
            if preprocessor.use_phonemizer and preprocessor.phonemizer:
                try:
                    phonemized = preprocessor._phonemize(test_text, "en")
                    logger.info(f"  Original: '{test_text}'")
                    logger.info(f"  Phonemized: '{phonemized}'")
                    logger.info(f"  Length change: {len(test_text)} -> {len(phonemized)}")
                except Exception as e:
                    logger.error(f"  Phonemization failed: {e}")
                    phonemized = test_text
            else:
                logger.info("  Phonemizer not available")
                phonemized = test_text
            
            # Test full preprocessing
            logger.info("Step 2: Full preprocessing test")
            try:
                processed = preprocessor.preprocess(test_text)
                logger.info(f"  Fully processed: '{processed}'")
            except Exception as e:
                logger.error(f"  Full preprocessing failed: {e}")
                processed = test_text
            
            # Test with phonemization enabled
            logger.info("Step 3: TTS with phonemization")
            original_setting = preprocessor.use_phonemizer
            preprocessor.use_phonemizer = True
            
            try:
                result_with = await inference_engine.generate_speech(
                    text=test_text,
                    voice="alloy",
                    response_format="wav"
                )
                
                # Quick transcription test
                audio_data, _ = await audio_processor.process_audio(
                    audio_array=result_with["audio_data"],
                    sample_rate=result_with["sample_rate"],
                    output_format="wav"
                )
                
                validation_result = whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=audio_data,
                    sample_rate=result_with["sample_rate"]
                )
                
                accuracy_with = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                transcription_with = validation_result.get("transcription", "")
                
                logger.info(f"  With phonemization:")
                logger.info(f"    Transcription: '{transcription_with}'")
                logger.info(f"    Accuracy: {accuracy_with:.1f}%")
                logger.info(f"    RTF: {result_with.get('rtf', 0):.3f}")
                
            except Exception as e:
                logger.error(f"  TTS with phonemization failed: {e}")
                accuracy_with = 0
                transcription_with = ""
            
            # Test without phonemization
            logger.info("Step 4: TTS without phonemization")
            preprocessor.use_phonemizer = False
            
            try:
                result_without = await inference_engine.generate_speech(
                    text=test_text,
                    voice="alloy",
                    response_format="wav"
                )
                
                # Quick transcription test
                audio_data, _ = await audio_processor.process_audio(
                    audio_array=result_without["audio_data"],
                    sample_rate=result_without["sample_rate"],
                    output_format="wav"
                )
                
                validation_result = whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=audio_data,
                    sample_rate=result_without["sample_rate"]
                )
                
                accuracy_without = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                transcription_without = validation_result.get("transcription", "")
                
                logger.info(f"  Without phonemization:")
                logger.info(f"    Transcription: '{transcription_without}'")
                logger.info(f"    Accuracy: {accuracy_without:.1f}%")
                logger.info(f"    RTF: {result_without.get('rtf', 0):.3f}")
                
            except Exception as e:
                logger.error(f"  TTS without phonemization failed: {e}")
                accuracy_without = 0
                transcription_without = ""
            
            # Restore original setting
            preprocessor.use_phonemizer = original_setting
            
            # Summary for this test case
            improvement = accuracy_without - accuracy_with
            logger.info(f"Step 5: Summary")
            logger.info(f"  Text: '{test_text}'")
            logger.info(f"  Improvement: {improvement:.1f} percentage points")
            
            if accuracy_without > 50:
                logger.info(f"  🎉 SUCCESS: Without phonemization is intelligible!")
                break
            elif improvement > 10:
                logger.info(f"  ✅ SIGNIFICANT IMPROVEMENT detected")
            elif improvement > 0:
                logger.info(f"  ✅ Minor improvement")
            else:
                logger.info(f"  ❌ No improvement")
        
        # Test direct SpeechT5 processor
        logger.info(f"\n{'='*60}")
        logger.info("DIRECT SPEECHT5 PROCESSOR TEST")
        logger.info(f"{'='*60}")
        
        try:
            # Get the model and test its processor directly
            model = await inference_engine._ensure_model_loaded()
            
            test_text = "Hello world"
            logger.info(f"Testing direct SpeechT5 processor with: '{test_text}'")
            
            # Test with raw text
            inputs = model.processor(text=test_text, return_tensors="pt")
            logger.info(f"SpeechT5 processor input_ids shape: {inputs['input_ids'].shape}")
            logger.info(f"SpeechT5 processor input_ids: {inputs['input_ids']}")
            
            # Test with phonemized text
            if preprocessor.use_phonemizer and preprocessor.phonemizer:
                phonemized_text = preprocessor._phonemize(test_text, "en")
                inputs_phonemized = model.processor(text=phonemized_text, return_tensors="pt")
                logger.info(f"Phonemized text: '{phonemized_text}'")
                logger.info(f"Phonemized input_ids shape: {inputs_phonemized['input_ids'].shape}")
                logger.info(f"Phonemized input_ids: {inputs_phonemized['input_ids']}")
                
                # Compare token sequences
                if not inputs['input_ids'].equal(inputs_phonemized['input_ids']):
                    logger.info("🔍 CRITICAL: Raw text and phonemized text produce different tokens!")
                    logger.info("This confirms the phonemization is changing the input to SpeechT5")
                else:
                    logger.info("Raw text and phonemized text produce identical tokens")
            
        except Exception as e:
            logger.error(f"Direct processor test failed: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info("DEBUGGING COMPLETE")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Debugging failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main execution function."""
    await debug_phonemization()


if __name__ == "__main__":
    asyncio.run(main())
