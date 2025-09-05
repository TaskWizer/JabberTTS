#!/usr/bin/env python3
"""
Test Phonemization Fix

This script tests the permanent fix for the SpeechT5 phonemization issue.

Usage:
    python jabbertts/scripts/test_phonemization_fix.py
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


async def test_fix():
    """Test the phonemization fix."""
    logger.info("🧪 Testing Permanent Phonemization Fix")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        whisper_validator = get_whisper_validator("base")
        
        # Test cases
        test_cases = [
            "Hello",
            "Hello world",
            "Hello world.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing one two three four five."
        ]
        
        results = []
        
        for i, test_text in enumerate(test_cases, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"TEST CASE {i}: '{test_text}'")
            logger.info(f"{'='*50}")
            
            try:
                # Generate speech with the fix
                result = await inference_engine.generate_speech(
                    text=test_text,
                    voice="alloy",
                    response_format="wav"
                )
                
                # Process audio
                audio_data, _ = await audio_processor.process_audio(
                    audio_array=result["audio_data"],
                    sample_rate=result["sample_rate"],
                    output_format="wav"
                )
                
                # Transcribe with Whisper
                validation_result = whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=audio_data,
                    sample_rate=result["sample_rate"]
                )
                
                # Extract metrics
                accuracy_metrics = validation_result.get("accuracy_metrics", {})
                accuracy = accuracy_metrics.get("overall_accuracy", 0)
                wer = accuracy_metrics.get("wer", 1.0)
                cer = accuracy_metrics.get("cer", 1.0)
                transcription = validation_result.get("transcription", "")
                
                # Log results
                logger.info(f"Original: '{test_text}'")
                logger.info(f"Transcribed: '{transcription}'")
                logger.info(f"Accuracy: {accuracy:.1f}%")
                logger.info(f"WER: {wer:.3f}")
                logger.info(f"CER: {cer:.3f}")
                logger.info(f"RTF: {result.get('rtf', 0):.3f}")
                
                # Determine status
                if accuracy >= 95:
                    status = "🟢 EXCELLENT"
                elif accuracy >= 80:
                    status = "🟡 GOOD"
                elif accuracy >= 50:
                    status = "🟠 POOR"
                else:
                    status = "🔴 UNINTELLIGIBLE"
                
                logger.info(f"Status: {status}")
                
                # Store results
                results.append({
                    "text": test_text,
                    "transcription": transcription,
                    "accuracy": accuracy,
                    "wer": wer,
                    "cer": cer,
                    "rtf": result.get("rtf", 0),
                    "status": status
                })
                
                # Check for breakthrough
                if accuracy >= 50:
                    logger.info("🎉 BREAKTHROUGH: Intelligible audio achieved!")
                    break
                    
            except Exception as e:
                logger.error(f"Test case {i} failed: {e}")
                results.append({
                    "text": test_text,
                    "error": str(e),
                    "accuracy": 0,
                    "status": "🔴 ERROR"
                })
        
        # Overall analysis
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL RESULTS")
        logger.info(f"{'='*60}")
        
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            avg_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results)
            max_accuracy = max(r["accuracy"] for r in successful_results)
            best_result = max(successful_results, key=lambda x: x["accuracy"])
            
            logger.info(f"Tests completed: {len(successful_results)}")
            logger.info(f"Average accuracy: {avg_accuracy:.1f}%")
            logger.info(f"Maximum accuracy: {max_accuracy:.1f}%")
            logger.info(f"Best result: '{best_result['text']}' → '{best_result['transcription']}'")
            
            # Determine fix effectiveness
            if max_accuracy >= 95:
                logger.info("🎉 FIX SUCCESSFUL: Excellent intelligibility achieved!")
                fix_status = "SUCCESSFUL"
            elif max_accuracy >= 80:
                logger.info("✅ FIX EFFECTIVE: Good intelligibility achieved!")
                fix_status = "EFFECTIVE"
            elif max_accuracy >= 50:
                logger.info("✅ FIX HELPFUL: Partial intelligibility achieved!")
                fix_status = "HELPFUL"
            elif avg_accuracy > 5:
                logger.info("✅ FIX IMPROVES: Some improvement detected!")
                fix_status = "IMPROVES"
            else:
                logger.info("❌ FIX INSUFFICIENT: Still unintelligible")
                fix_status = "INSUFFICIENT"
        else:
            logger.error("❌ ALL TESTS FAILED")
            fix_status = "FAILED"
            avg_accuracy = 0
            max_accuracy = 0
        
        # Save results
        test_results = {
            "phonemization_fix_test": {
                "timestamp": datetime.now().isoformat(),
                "fix_status": fix_status,
                "summary": {
                    "tests_completed": len(successful_results),
                    "average_accuracy": avg_accuracy,
                    "maximum_accuracy": max_accuracy
                },
                "individual_results": results
            }
        }
        
        output_file = Path("temp") / "phonemization_fix_test_results.json"
        with open(output_file, "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {output_file}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main execution function."""
    logger.info("🔧 TESTING PERMANENT PHONEMIZATION FIX")
    logger.info("=" * 60)
    logger.info("This test validates the fix that disables phonemization for SpeechT5")
    logger.info("Expected: Significant improvement in transcription accuracy")
    logger.info("=" * 60)
    
    results = await test_fix()
    
    if results:
        fix_status = results["phonemization_fix_test"]["fix_status"]
        max_accuracy = results["phonemization_fix_test"]["summary"]["maximum_accuracy"]
        
        logger.info(f"\n🎯 FINAL RESULT: {fix_status}")
        logger.info(f"Maximum accuracy achieved: {max_accuracy:.1f}%")
        
        if fix_status in ["SUCCESSFUL", "EFFECTIVE", "HELPFUL"]:
            logger.info("✅ The phonemization fix is working!")
        else:
            logger.info("❌ Additional investigation needed")
    else:
        logger.error("❌ Test execution failed")


if __name__ == "__main__":
    asyncio.run(main())
