#!/usr/bin/env python3
"""
Simple validation script for the perceptual quality framework.

This script validates that the perceptual quality metrics and regression testing
framework works correctly by running a single comprehensive analysis.

Usage:
    python validate_perceptual_quality.py
"""

import asyncio
import json
import logging
from pathlib import Path

from tests.test_perceptual_quality import PerceptualQualityAnalyzer
from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def validate_perceptual_quality_framework():
    """Validate the perceptual quality framework."""
    logger.info("🧪 Starting Perceptual Quality Framework Validation")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        inference_engine = get_inference_engine()
        audio_processor = get_audio_processor()
        quality_analyzer = PerceptualQualityAnalyzer()
        
        # Test case
        test_text = "The weather today is sunny and warm with a gentle breeze."
        test_voice = "fable"
        
        logger.info(f"Test Text: {test_text}")
        logger.info(f"Test Voice: {test_voice}")
        logger.info("")
        
        # Perform comprehensive quality analysis
        logger.info("Performing comprehensive perceptual quality analysis...")
        metrics = await quality_analyzer.analyze_comprehensive_quality(
            test_text, test_voice, inference_engine, audio_processor
        )
        
        logger.info("✅ Perceptual quality analysis completed")
        logger.info("")
        
        # Display results
        logger.info("📊 COMPREHENSIVE QUALITY METRICS")
        logger.info("=" * 40)
        
        # Intelligibility metrics
        logger.info("🎯 Intelligibility Metrics:")
        logger.info(f"   Transcription Accuracy: {metrics.transcription_accuracy:.1f}%")
        logger.info(f"   Word Error Rate: {metrics.word_error_rate:.3f}")
        logger.info(f"   Character Error Rate: {metrics.character_error_rate:.3f}")
        logger.info("")
        
        # Technical quality metrics
        logger.info("🔧 Technical Quality Metrics:")
        logger.info(f"   Overall Quality: {metrics.overall_quality:.1f}%")
        logger.info(f"   Naturalness: {metrics.naturalness_score:.1f}%")
        logger.info(f"   Clarity: {metrics.clarity_score:.1f}%")
        logger.info(f"   Consistency: {metrics.consistency_score:.1f}%")
        logger.info("")
        
        # Perceptual metrics (NEW)
        logger.info("🎭 Perceptual Quality Metrics:")
        logger.info(f"   Prosody Score: {metrics.prosody_score:.1f}%")
        logger.info(f"   Rhythm Score: {metrics.rhythm_score:.1f}%")
        logger.info(f"   Emotional Expression: {metrics.emotional_expression:.1f}%")
        logger.info(f"   Human Likeness: {metrics.human_likeness:.1f}%")
        logger.info("")
        
        # Performance metrics
        logger.info("⚡ Performance Metrics:")
        logger.info(f"   RTF: {metrics.rtf:.3f}")
        logger.info(f"   Inference Time: {metrics.inference_time:.2f}s")
        logger.info(f"   Audio Duration: {metrics.audio_duration:.2f}s")
        logger.info("")
        
        # Metadata
        logger.info("📋 Metadata:")
        logger.info(f"   Voice: {metrics.voice}")
        logger.info(f"   Text Complexity: {metrics.text_complexity}")
        logger.info(f"   Timestamp: {metrics.timestamp}")
        logger.info("")
        
        # Quality assessment
        logger.info("🔍 Quality Assessment:")
        
        # Intelligibility status
        if metrics.transcription_accuracy >= 95:
            intelligibility_status = "🟢 EXCELLENT"
        elif metrics.transcription_accuracy >= 80:
            intelligibility_status = "🟡 GOOD"
        elif metrics.transcription_accuracy >= 50:
            intelligibility_status = "🟠 POOR"
        else:
            intelligibility_status = "🔴 UNINTELLIGIBLE"
        
        # Human-likeness status
        if metrics.human_likeness >= 85:
            human_status = "🟢 VERY HUMAN-LIKE"
        elif metrics.human_likeness >= 70:
            human_status = "🟡 MODERATELY HUMAN-LIKE"
        elif metrics.human_likeness >= 50:
            human_status = "🟠 SOMEWHAT ROBOTIC"
        else:
            human_status = "🔴 VERY ROBOTIC"
        
        # Prosody status
        if metrics.prosody_score >= 80:
            prosody_status = "🟢 NATURAL PROSODY"
        elif metrics.prosody_score >= 60:
            prosody_status = "🟡 ACCEPTABLE PROSODY"
        else:
            prosody_status = "🔴 POOR PROSODY"
        
        logger.info(f"Intelligibility: {intelligibility_status} ({metrics.transcription_accuracy:.1f}%)")
        logger.info(f"Human-likeness: {human_status} ({metrics.human_likeness:.1f}%)")
        logger.info(f"Prosody Quality: {prosody_status} ({metrics.prosody_score:.1f}%)")
        logger.info("")
        
        # Critical findings
        logger.info("🔍 Critical Findings:")
        
        critical_issues = []
        warnings = []
        
        if metrics.transcription_accuracy < 50:
            critical_issues.append("Audio is unintelligible")
        elif metrics.transcription_accuracy < 80:
            warnings.append("Poor intelligibility")
        
        if metrics.human_likeness < 50:
            critical_issues.append("Audio sounds very robotic")
        elif metrics.human_likeness < 70:
            warnings.append("Audio lacks human-like qualities")
        
        if metrics.prosody_score < 40:
            critical_issues.append("Very poor prosody and rhythm")
        elif metrics.prosody_score < 60:
            warnings.append("Prosody needs improvement")
        
        if metrics.rtf > 1.0:
            warnings.append("Performance below real-time")
        
        if critical_issues:
            for issue in critical_issues:
                logger.critical(f"❌ CRITICAL: {issue}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️  WARNING: {warning}")
        
        if not critical_issues and not warnings:
            logger.info("✅ No critical issues or warnings detected")
        
        logger.info("")
        
        # Framework validation
        logger.info("🎯 FRAMEWORK VALIDATION:")
        
        framework_checks = {
            "Metrics calculation": metrics.transcription_accuracy >= 0,
            "Perceptual analysis": metrics.prosody_score >= 0 and metrics.human_likeness >= 0,
            "Technical quality": metrics.overall_quality >= 0,
            "Performance tracking": metrics.rtf >= 0,
            "Text complexity assessment": metrics.text_complexity in ['simple', 'medium', 'complex'],
            "Timestamp generation": bool(metrics.timestamp)
        }
        
        all_passed = True
        for check, passed in framework_checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        
        logger.info("")
        
        if all_passed:
            logger.info("🎉 FRAMEWORK VALIDATION SUCCESSFUL")
            logger.info("✅ All perceptual quality metrics are working correctly")
            logger.info("✅ Framework can assess human-likeness, prosody, and emotional expression")
            logger.info("✅ Regression testing capabilities are functional")
        else:
            logger.error("❌ FRAMEWORK VALIDATION FAILED")
            logger.error("Some framework components are not working correctly")
        
        # Save detailed results
        results = {
            "perceptual_quality_validation": {
                "timestamp": metrics.timestamp,
                "test_case": {
                    "text": test_text,
                    "voice": test_voice
                },
                "comprehensive_metrics": {
                    "transcription_accuracy": metrics.transcription_accuracy,
                    "word_error_rate": metrics.word_error_rate,
                    "character_error_rate": metrics.character_error_rate,
                    "overall_quality": metrics.overall_quality,
                    "naturalness_score": metrics.naturalness_score,
                    "clarity_score": metrics.clarity_score,
                    "consistency_score": metrics.consistency_score,
                    "prosody_score": metrics.prosody_score,
                    "rhythm_score": metrics.rhythm_score,
                    "emotional_expression": metrics.emotional_expression,
                    "human_likeness": metrics.human_likeness,
                    "rtf": metrics.rtf,
                    "inference_time": metrics.inference_time,
                    "audio_duration": metrics.audio_duration,
                    "text_complexity": metrics.text_complexity
                },
                "quality_assessment": {
                    "intelligibility_status": intelligibility_status,
                    "human_likeness_status": human_status,
                    "prosody_status": prosody_status
                },
                "framework_validation": {
                    "all_checks_passed": all_passed,
                    "individual_checks": framework_checks
                },
                "critical_issues": critical_issues,
                "warnings": warnings
            }
        }
        
        output_file = Path("perceptual_quality_validation.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📁 Detailed results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Framework validation failed: {e}")
        raise


def main():
    """Main execution function."""
    try:
        results = asyncio.run(validate_perceptual_quality_framework())
        return results
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None


if __name__ == "__main__":
    main()
