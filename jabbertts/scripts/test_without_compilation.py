#!/usr/bin/env python3
"""
Test Without Torch Compilation

This script tests the SpeechT5 model without torch compilation to see
if compilation is causing the intelligibility issue.

Usage:
    python jabbertts/scripts/test_without_compilation.py
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import soundfile as sf

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.models.manager import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_without_compilation():
    """Test SpeechT5 without torch compilation."""
    logger.info("🧪 Testing SpeechT5 Without Torch Compilation")
    logger.info("=" * 60)
    logger.info("This test disables torch.compile to check if compilation")
    logger.info("is causing the intelligibility issue.")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        whisper_validator = get_whisper_validator("base")
        model_manager = get_model_manager()
        
        # Test cases
        test_cases = [
            "Hello",
            "Hello world",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        results = {
            "compilation_test": {
                "timestamp": datetime.now().isoformat(),
                "compiled_results": [],
                "non_compiled_results": [],
                "comparison": {}
            }
        }
        
        # Phase 1: Test with compilation (current state)
        logger.info("🔧 PHASE 1: Testing with torch.compile ENABLED")
        logger.info("=" * 50)
        
        for i, test_text in enumerate(test_cases, 1):
            logger.info(f"Test {i}: '{test_text}'")
            
            try:
                # Generate with compilation
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
                
                # Transcribe
                validation_result = whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=audio_data,
                    sample_rate=result["sample_rate"]
                )
                
                accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                transcription = validation_result.get("transcription", "")
                
                compiled_result = {
                    "text": test_text,
                    "transcription": transcription,
                    "accuracy": accuracy,
                    "rtf": result.get("rtf", 0),
                    "audio_duration": len(result["audio_data"]) / result["sample_rate"]
                }
                
                results["compilation_test"]["compiled_results"].append(compiled_result)
                
                logger.info(f"  Compiled - Transcription: '{transcription}'")
                logger.info(f"  Compiled - Accuracy: {accuracy:.1f}%")
                logger.info(f"  Compiled - RTF: {result.get('rtf', 0):.3f}")
                
            except Exception as e:
                logger.error(f"  Compiled test failed: {e}")
                results["compilation_test"]["compiled_results"].append({
                    "text": test_text,
                    "error": str(e)
                })
        
        # Phase 2: Test without compilation
        logger.info("\n🔧 PHASE 2: Testing with torch.compile DISABLED")
        logger.info("=" * 50)
        
        # Unload current model
        logger.info("Unloading compiled model...")
        model_manager.unload_all_models()
        
        # Temporarily disable torch.compile
        logger.info("Disabling torch.compile...")
        original_compile = torch.compile
        torch.compile = lambda model, *args, **kwargs: model  # No-op compile
        
        try:
            for i, test_text in enumerate(test_cases, 1):
                logger.info(f"Test {i}: '{test_text}'")
                
                try:
                    # Generate without compilation
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
                    
                    # Transcribe
                    validation_result = whisper_validator.validate_tts_output(
                        original_text=test_text,
                        audio_data=audio_data,
                        sample_rate=result["sample_rate"]
                    )
                    
                    accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                    transcription = validation_result.get("transcription", "")
                    
                    non_compiled_result = {
                        "text": test_text,
                        "transcription": transcription,
                        "accuracy": accuracy,
                        "rtf": result.get("rtf", 0),
                        "audio_duration": len(result["audio_data"]) / result["sample_rate"]
                    }
                    
                    results["compilation_test"]["non_compiled_results"].append(non_compiled_result)
                    
                    logger.info(f"  Non-compiled - Transcription: '{transcription}'")
                    logger.info(f"  Non-compiled - Accuracy: {accuracy:.1f}%")
                    logger.info(f"  Non-compiled - RTF: {result.get('rtf', 0):.3f}")
                    
                    # Save audio sample for comparison
                    audio_file = Path("temp") / f"non_compiled_{i}.wav"
                    sf.write(str(audio_file), result["audio_data"], result["sample_rate"])
                    logger.info(f"  Non-compiled audio saved: {audio_file}")
                    
                except Exception as e:
                    logger.error(f"  Non-compiled test failed: {e}")
                    results["compilation_test"]["non_compiled_results"].append({
                        "text": test_text,
                        "error": str(e)
                    })
        
        finally:
            # Restore original torch.compile
            logger.info("Restoring torch.compile...")
            torch.compile = original_compile
        
        # Phase 3: Compare results
        logger.info("\n🔍 PHASE 3: COMPARISON ANALYSIS")
        logger.info("=" * 50)
        
        compiled_results = [r for r in results["compilation_test"]["compiled_results"] if "error" not in r]
        non_compiled_results = [r for r in results["compilation_test"]["non_compiled_results"] if "error" not in r]
        
        if compiled_results and non_compiled_results:
            # Calculate averages
            avg_compiled_accuracy = sum(r["accuracy"] for r in compiled_results) / len(compiled_results)
            avg_non_compiled_accuracy = sum(r["accuracy"] for r in non_compiled_results) / len(non_compiled_results)
            
            avg_compiled_rtf = sum(r["rtf"] for r in compiled_results) / len(compiled_results)
            avg_non_compiled_rtf = sum(r["rtf"] for r in non_compiled_results) / len(non_compiled_results)
            
            accuracy_improvement = avg_non_compiled_accuracy - avg_compiled_accuracy
            rtf_change = avg_non_compiled_rtf - avg_compiled_rtf
            
            # Find best results
            best_compiled = max(compiled_results, key=lambda x: x["accuracy"])
            best_non_compiled = max(non_compiled_results, key=lambda x: x["accuracy"])
            
            # Store comparison
            comparison = {
                "avg_compiled_accuracy": avg_compiled_accuracy,
                "avg_non_compiled_accuracy": avg_non_compiled_accuracy,
                "accuracy_improvement": accuracy_improvement,
                "avg_compiled_rtf": avg_compiled_rtf,
                "avg_non_compiled_rtf": avg_non_compiled_rtf,
                "rtf_change": rtf_change,
                "best_compiled_accuracy": best_compiled["accuracy"],
                "best_non_compiled_accuracy": best_non_compiled["accuracy"],
                "compilation_is_issue": accuracy_improvement > 10
            }
            
            results["compilation_test"]["comparison"] = comparison
            
            # Log comparison
            logger.info(f"Average Accuracy:")
            logger.info(f"  With compilation: {avg_compiled_accuracy:.1f}%")
            logger.info(f"  Without compilation: {avg_non_compiled_accuracy:.1f}%")
            logger.info(f"  Improvement: {accuracy_improvement:.1f} percentage points")
            logger.info("")
            logger.info(f"Best Results:")
            logger.info(f"  Compiled: {best_compiled['accuracy']:.1f}% - '{best_compiled['transcription']}'")
            logger.info(f"  Non-compiled: {best_non_compiled['accuracy']:.1f}% - '{best_non_compiled['transcription']}'")
            logger.info("")
            logger.info(f"Performance (RTF):")
            logger.info(f"  With compilation: {avg_compiled_rtf:.3f}")
            logger.info(f"  Without compilation: {avg_non_compiled_rtf:.3f}")
            logger.info(f"  Change: {rtf_change:.3f}")
            logger.info("")
            
            # Determine conclusion
            if accuracy_improvement > 50:
                conclusion = "🎉 MAJOR BREAKTHROUGH: Disabling compilation dramatically improves intelligibility!"
                fix_status = "MAJOR_BREAKTHROUGH"
            elif accuracy_improvement > 20:
                conclusion = "✅ SIGNIFICANT IMPROVEMENT: Disabling compilation significantly improves intelligibility!"
                fix_status = "SIGNIFICANT_IMPROVEMENT"
            elif accuracy_improvement > 5:
                conclusion = "✅ IMPROVEMENT: Disabling compilation improves intelligibility!"
                fix_status = "IMPROVEMENT"
            elif accuracy_improvement > 0:
                conclusion = "✅ Minor improvement from disabling compilation"
                fix_status = "MINOR_IMPROVEMENT"
            else:
                conclusion = "❌ No improvement from disabling compilation"
                fix_status = "NO_IMPROVEMENT"
            
            logger.info(f"🎯 CONCLUSION: {conclusion}")
            
            if best_non_compiled["accuracy"] > 80:
                logger.info("🎉 INTELLIGIBLE AUDIO ACHIEVED!")
            elif best_non_compiled["accuracy"] > 50:
                logger.info("✅ Partially intelligible audio achieved!")
            
            comparison["conclusion"] = conclusion
            comparison["fix_status"] = fix_status
            
        else:
            logger.error("❌ Could not compare results due to test failures")
            comparison = {"error": "Comparison failed due to test errors"}
            results["compilation_test"]["comparison"] = comparison
        
        # Save results
        output_file = Path("temp") / "compilation_test_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main execution function."""
    logger.info("🚨 CRITICAL TEST: Torch Compilation Impact on Intelligibility")
    logger.info("=" * 70)
    logger.info("Testing hypothesis: torch.compile is causing unintelligible audio")
    logger.info("=" * 70)
    
    results = await test_without_compilation()
    
    if results and "comparison" in results["compilation_test"]:
        comparison = results["compilation_test"]["comparison"]
        
        if "fix_status" in comparison:
            fix_status = comparison["fix_status"]
            accuracy_improvement = comparison.get("accuracy_improvement", 0)
            
            logger.info(f"\n🎯 FINAL RESULT: {fix_status}")
            logger.info(f"Accuracy improvement: {accuracy_improvement:.1f} percentage points")
            
            if fix_status in ["MAJOR_BREAKTHROUGH", "SIGNIFICANT_IMPROVEMENT"]:
                logger.info("✅ TORCH COMPILATION IS THE ROOT CAUSE!")
                logger.info("Recommendation: Disable torch.compile for SpeechT5")
            elif fix_status == "IMPROVEMENT":
                logger.info("✅ Torch compilation contributes to the issue")
                logger.info("Recommendation: Consider disabling torch.compile")
            else:
                logger.info("❌ Torch compilation is not the main issue")
                logger.info("Additional investigation needed")
    else:
        logger.error("❌ Test failed or incomplete")


if __name__ == "__main__":
    asyncio.run(main())
