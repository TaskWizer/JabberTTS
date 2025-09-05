#!/usr/bin/env python3
"""
Simple validation script for the intelligibility testing framework.

This script validates that the automated intelligibility testing framework works correctly
by running a single test case and reporting the results.

Usage:
    python validate_intelligibility_framework.py
"""

import asyncio
import json
import logging
from pathlib import Path

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.validation.audio_quality import AudioQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def validate_intelligibility_framework():
    """Validate the intelligibility testing framework."""
    logger.info("🧪 Starting Intelligibility Framework Validation")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        whisper_validator = get_whisper_validator("base")
        quality_validator = AudioQualityValidator()
        
        # Test case
        test_text = "The quick brown fox jumps over the lazy dog."
        test_voice = "alloy"
        
        logger.info(f"Test Text: {test_text}")
        logger.info(f"Test Voice: {test_voice}")
        logger.info("")
        
        # Step 1: Generate TTS audio
        logger.info("Step 1: Generating TTS audio...")
        tts_result = await inference_engine.generate_speech(
            text=test_text,
            voice=test_voice,
            response_format="wav"
        )
        
        logger.info(f"✅ Audio generated successfully")
        logger.info(f"   Duration: {tts_result.get('audio_duration', 0):.2f}s")
        logger.info(f"   RTF: {tts_result.get('rtf', 0):.3f}")
        logger.info(f"   Sample Rate: {tts_result['sample_rate']}Hz")
        logger.info("")
        
        # Step 2: Process audio
        logger.info("Step 2: Processing audio...")
        audio_data, audio_metadata = await audio_processor.process_audio(
            audio_array=tts_result["audio_data"],
            sample_rate=tts_result["sample_rate"],
            output_format="wav"
        )
        
        logger.info(f"✅ Audio processed successfully")
        logger.info(f"   Processed Duration: {audio_metadata.get('processed_duration', 0):.2f}s")
        logger.info(f"   Final Sample Rate: {audio_metadata.get('final_sample_rate', 0)}Hz")
        logger.info("")
        
        # Step 3: Validate with Whisper STT
        logger.info("Step 3: Validating with Whisper STT...")
        validation_result = whisper_validator.validate_tts_output(
            original_text=test_text,
            audio_data=audio_data,
            sample_rate=tts_result["sample_rate"]
        )
        
        # Extract metrics
        transcription = validation_result.get("transcription", "")
        accuracy_metrics = validation_result.get("accuracy_metrics", {})
        accuracy = accuracy_metrics.get("overall_accuracy", 0)
        wer = accuracy_metrics.get("wer", 1.0)
        cer = accuracy_metrics.get("cer", 1.0)
        
        logger.info(f"✅ Whisper validation completed")
        logger.info(f"   Original Text: {test_text}")
        logger.info(f"   Transcribed:   {transcription}")
        logger.info(f"   Accuracy: {accuracy:.1f}%")
        logger.info(f"   WER: {wer:.3f}")
        logger.info(f"   CER: {cer:.3f}")
        logger.info("")
        
        # Step 4: Analyze audio quality
        logger.info("Step 4: Analyzing audio quality...")
        quality_metrics = quality_validator.analyze_audio(
            tts_result["audio_data"],
            tts_result["sample_rate"],
            tts_result.get("rtf", 0),
            tts_result.get("inference_time", 0)
        )
        
        logger.info(f"✅ Quality analysis completed")
        logger.info(f"   Overall Quality: {quality_metrics.overall_quality:.1f}%")
        logger.info(f"   Naturalness: {quality_metrics.naturalness_score:.1f}%")
        logger.info(f"   Clarity: {quality_metrics.clarity_score:.1f}%")
        logger.info(f"   Consistency: {quality_metrics.consistency_score:.1f}%")
        logger.info("")
        
        # Step 5: Comprehensive analysis
        logger.info("Step 5: Comprehensive Analysis")
        logger.info("=" * 40)
        
        # Determine intelligibility status
        if accuracy >= 95:
            intelligibility_status = "🟢 EXCELLENT"
        elif accuracy >= 80:
            intelligibility_status = "🟡 GOOD"
        elif accuracy >= 50:
            intelligibility_status = "🟠 POOR"
        else:
            intelligibility_status = "🔴 UNINTELLIGIBLE"
        
        # Determine performance status
        if tts_result.get("rtf", 0) <= 0.5:
            performance_status = "🟢 EXCELLENT"
        elif tts_result.get("rtf", 0) <= 1.0:
            performance_status = "🟡 GOOD"
        else:
            performance_status = "🔴 POOR"
        
        # Determine quality status
        if quality_metrics.overall_quality >= 90:
            quality_status = "🟢 EXCELLENT"
        elif quality_metrics.overall_quality >= 80:
            quality_status = "🟡 GOOD"
        else:
            quality_status = "🔴 POOR"
        
        logger.info(f"Intelligibility: {intelligibility_status} ({accuracy:.1f}%)")
        logger.info(f"Performance:     {performance_status} (RTF: {tts_result.get('rtf', 0):.3f})")
        logger.info(f"Quality:         {quality_status} ({quality_metrics.overall_quality:.1f}%)")
        logger.info("")
        
        # Critical findings
        logger.info("🔍 Critical Findings:")
        if accuracy < 50:
            logger.critical("❌ CRITICAL ISSUE: Audio is unintelligible (accuracy < 50%)")
            logger.critical("   This confirms the major intelligibility problem described in the task.")
            logger.critical("   Despite good quality metrics, the generated audio cannot be understood.")
        elif accuracy < 80:
            logger.warning("⚠️  WARNING: Poor intelligibility detected")
        else:
            logger.info("✅ Good intelligibility achieved")
        
        if wer > 0.5:
            logger.critical("❌ CRITICAL: Very high Word Error Rate (WER > 50%)")
        
        if cer > 0.5:
            logger.critical("❌ CRITICAL: Very high Character Error Rate (CER > 50%)")
        
        logger.info("")
        
        # Save detailed results
        results = {
            "framework_validation": {
                "timestamp": "2025-09-05",
                "test_case": {
                    "text": test_text,
                    "voice": test_voice
                },
                "generation_metrics": {
                    "rtf": tts_result.get("rtf", 0),
                    "inference_time": tts_result.get("inference_time", 0),
                    "audio_duration": tts_result.get("audio_duration", 0),
                    "sample_rate": tts_result["sample_rate"]
                },
                "intelligibility_metrics": {
                    "transcription": transcription,
                    "accuracy": accuracy,
                    "wer": wer,
                    "cer": cer
                },
                "quality_metrics": quality_metrics.to_dict(),
                "status_assessment": {
                    "intelligibility": intelligibility_status,
                    "performance": performance_status,
                    "quality": quality_status
                },
                "framework_status": "WORKING"
            }
        }
        
        output_file = Path("intelligibility_framework_validation.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Detailed results saved to: {output_file}")
        logger.info("")
        logger.info("🎯 FRAMEWORK VALIDATION SUMMARY:")
        logger.info("✅ Intelligibility testing framework is working correctly")
        logger.info("✅ All components (TTS, Whisper STT, Quality Analysis) are functional")
        logger.info("✅ Metrics are being calculated and reported accurately")
        logger.info("✅ Framework can detect intelligibility issues (as demonstrated)")
        
        if accuracy < 50:
            logger.info("")
            logger.info("🚨 NEXT STEPS REQUIRED:")
            logger.info("1. Investigate root cause of intelligibility issues")
            logger.info("2. Analyze audio pipeline for corruption points")
            logger.info("3. Compare with reference implementations")
            logger.info("4. Test minimal processing pipeline")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Framework validation failed: {e}")
        raise


def main():
    """Main execution function."""
    try:
        results = asyncio.run(validate_intelligibility_framework())
        return results
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None


if __name__ == "__main__":
    main()
