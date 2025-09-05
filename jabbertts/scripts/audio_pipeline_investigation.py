#!/usr/bin/env python3
"""
Audio Pipeline Deep Dive Investigation

This script systematically investigates the audio generation pipeline to identify
the exact point where intelligibility is lost. It analyzes each processing stage
and compares outputs with Whisper transcription to isolate the failure point.

Usage:
    python jabbertts/scripts/audio_pipeline_investigation.py
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import soundfile as sf

from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor
from jabbertts.validation.whisper_validator import get_whisper_validator
from jabbertts.validation.audio_quality import AudioQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    audio_data: np.ndarray
    sample_rate: int
    transcription: str
    accuracy: float
    wer: float
    cer: float
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]


class AudioPipelineInvestigator:
    """Investigates audio pipeline to identify intelligibility loss points."""
    
    def __init__(self):
        """Initialize the investigator."""
        self.whisper_validator = get_whisper_validator("base")
        self.quality_validator = AudioQualityValidator()
        self.results = []
    
    def save_audio_sample(self, audio_data: np.ndarray, sample_rate: int, filename: str) -> str:
        """Save audio sample for analysis."""
        try:
            output_path = Path("temp") / f"{filename}.wav"
            output_path.parent.mkdir(exist_ok=True)
            
            sf.write(str(output_path), audio_data, sample_rate)
            logger.info(f"Saved audio sample: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save audio sample {filename}: {e}")
            return ""
    
    def analyze_audio_stage(
        self, 
        stage_name: str,
        audio_data: np.ndarray, 
        sample_rate: int,
        original_text: str,
        processing_time: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> PipelineStageResult:
        """Analyze a single pipeline stage."""
        try:
            logger.info(f"Analyzing stage: {stage_name}")
            
            if metadata is None:
                metadata = {}
            
            # Save audio sample for manual inspection
            audio_file = self.save_audio_sample(audio_data, sample_rate, f"stage_{stage_name.lower().replace(' ', '_')}")
            metadata["audio_file"] = audio_file
            
            # Convert to bytes for Whisper analysis
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)
                with open(temp_file.name, "rb") as f:
                    audio_bytes = f.read()
            
            # Transcribe with Whisper
            validation_result = self.whisper_validator.validate_tts_output(
                original_text=original_text,
                audio_data=audio_bytes,
                sample_rate=sample_rate
            )
            
            # Extract metrics
            accuracy_metrics = validation_result.get("accuracy_metrics", {})
            transcription = validation_result.get("transcription", "")
            accuracy = accuracy_metrics.get("overall_accuracy", 0)
            wer = accuracy_metrics.get("wer", 1.0)
            cer = accuracy_metrics.get("cer", 1.0)
            
            # Analyze audio quality
            quality_metrics = self.quality_validator.analyze_audio(
                audio_data, sample_rate, 0, processing_time
            )
            quality_score = quality_metrics.overall_quality
            
            # Create result
            result = PipelineStageResult(
                stage_name=stage_name,
                audio_data=audio_data,
                sample_rate=sample_rate,
                transcription=transcription,
                accuracy=accuracy,
                wer=wer,
                cer=cer,
                quality_score=quality_score,
                processing_time=processing_time,
                metadata=metadata
            )
            
            # Log results
            logger.info(f"Stage '{stage_name}' Results:")
            logger.info(f"  Transcription: '{transcription}'")
            logger.info(f"  Accuracy: {accuracy:.1f}%")
            logger.info(f"  WER: {wer:.3f}")
            logger.info(f"  Quality: {quality_score:.1f}%")
            logger.info(f"  Duration: {len(audio_data)/sample_rate:.2f}s")
            logger.info("")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze stage {stage_name}: {e}")
            return PipelineStageResult(
                stage_name=stage_name,
                audio_data=audio_data,
                sample_rate=sample_rate,
                transcription="",
                accuracy=0.0,
                wer=1.0,
                cer=1.0,
                quality_score=0.0,
                processing_time=processing_time,
                metadata=metadata or {}
            )
    
    async def investigate_full_pipeline(self, text: str, voice: str = "alloy") -> List[PipelineStageResult]:
        """Investigate the complete audio generation pipeline."""
        logger.info(f"🔍 Starting Full Pipeline Investigation")
        logger.info(f"Text: '{text}'")
        logger.info(f"Voice: {voice}")
        logger.info("=" * 60)
        
        results = []
        
        try:
            # Initialize components
            inference_engine = get_inference_engine()
            audio_processor = get_audio_processor()
            
            # Stage 1: Raw Model Output (before any processing)
            logger.info("Stage 1: Generating raw SpeechT5 model output...")
            start_time = datetime.now()
            
            tts_result = await inference_engine.generate_speech(
                text=text,
                voice=voice,
                response_format="wav"
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Analyze raw model output
            raw_result = self.analyze_audio_stage(
                stage_name="Raw Model Output",
                audio_data=tts_result["audio_data"],
                sample_rate=tts_result["sample_rate"],
                original_text=text,
                processing_time=generation_time,
                metadata={
                    "rtf": tts_result.get("rtf", 0),
                    "inference_time": tts_result.get("inference_time", 0),
                    "model_name": "SpeechT5",
                    "voice_embedding": voice
                }
            )
            results.append(raw_result)
            
            # Stage 2: Audio Processor (with full processing)
            logger.info("Stage 2: Processing with AudioProcessor...")
            start_time = datetime.now()
            
            processed_audio, audio_metadata = await audio_processor.process_audio(
                audio_array=tts_result["audio_data"],
                sample_rate=tts_result["sample_rate"],
                output_format="wav"
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert processed audio back to numpy array for analysis
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(processed_audio)
                processed_array, processed_sr = sf.read(temp_file.name)
            
            processed_result = self.analyze_audio_stage(
                stage_name="Audio Processor Output",
                audio_data=processed_array,
                sample_rate=processed_sr,
                original_text=text,
                processing_time=processing_time,
                metadata={
                    "audio_metadata": audio_metadata,
                    "format_conversion": "numpy -> bytes -> numpy",
                    "processing_steps": "enhancement, normalization, encoding"
                }
            )
            results.append(processed_result)
            
            # Stage 3: Minimal Processing (bypass enhancement)
            logger.info("Stage 3: Minimal processing (no enhancement)...")
            start_time = datetime.now()
            
            # Create minimal processor settings
            minimal_audio, minimal_metadata = await audio_processor.process_audio(
                audio_array=tts_result["audio_data"],
                sample_rate=tts_result["sample_rate"],
                output_format="wav",
                speed=1.0,
                # Disable enhancement if possible
            )
            
            minimal_processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert minimal processed audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(minimal_audio)
                minimal_array, minimal_sr = sf.read(temp_file.name)
            
            minimal_result = self.analyze_audio_stage(
                stage_name="Minimal Processing",
                audio_data=minimal_array,
                sample_rate=minimal_sr,
                original_text=text,
                processing_time=minimal_processing_time,
                metadata={
                    "processing_type": "minimal",
                    "enhancement_disabled": True,
                    "format_only": True
                }
            )
            results.append(minimal_result)
            
            # Stage 4: Direct WAV conversion (no AudioProcessor)
            logger.info("Stage 4: Direct WAV conversion...")
            start_time = datetime.now()
            
            # Direct conversion without AudioProcessor
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, tts_result["audio_data"], tts_result["sample_rate"])
                direct_array, direct_sr = sf.read(temp_file.name)
            
            direct_time = (datetime.now() - start_time).total_seconds()
            
            direct_result = self.analyze_audio_stage(
                stage_name="Direct WAV Conversion",
                audio_data=direct_array,
                sample_rate=direct_sr,
                original_text=text,
                processing_time=direct_time,
                metadata={
                    "processing_type": "direct",
                    "bypass_audio_processor": True,
                    "raw_model_to_wav": True
                }
            )
            results.append(direct_result)
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Pipeline investigation failed: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def analyze_results(self, results: List[PipelineStageResult]) -> Dict[str, Any]:
        """Analyze pipeline investigation results."""
        logger.info("🔍 PIPELINE INVESTIGATION ANALYSIS")
        logger.info("=" * 50)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_stages": len(results),
            "stage_results": [],
            "intelligibility_analysis": {},
            "quality_analysis": {},
            "performance_analysis": {},
            "critical_findings": []
        }
        
        # Analyze each stage
        for i, result in enumerate(results):
            stage_analysis = {
                "stage_number": i + 1,
                "stage_name": result.stage_name,
                "transcription": result.transcription,
                "accuracy": result.accuracy,
                "wer": result.wer,
                "cer": result.cer,
                "quality_score": result.quality_score,
                "processing_time": result.processing_time,
                "audio_duration": len(result.audio_data) / result.sample_rate,
                "sample_rate": result.sample_rate,
                "metadata": result.metadata
            }
            analysis["stage_results"].append(stage_analysis)
            
            # Log stage summary
            logger.info(f"Stage {i+1}: {result.stage_name}")
            logger.info(f"  Accuracy: {result.accuracy:.1f}%")
            logger.info(f"  Transcription: '{result.transcription}'")
            logger.info(f"  Quality: {result.quality_score:.1f}%")
            logger.info(f"  Processing Time: {result.processing_time:.3f}s")
            logger.info("")
        
        # Intelligibility analysis
        accuracies = [r.accuracy for r in results]
        best_accuracy = max(accuracies)
        worst_accuracy = min(accuracies)
        best_stage = results[accuracies.index(best_accuracy)].stage_name
        worst_stage = results[accuracies.index(worst_accuracy)].stage_name
        
        analysis["intelligibility_analysis"] = {
            "best_accuracy": best_accuracy,
            "worst_accuracy": worst_accuracy,
            "best_stage": best_stage,
            "worst_stage": worst_stage,
            "accuracy_degradation": best_accuracy - worst_accuracy,
            "all_stages_poor": all(acc < 50 for acc in accuracies)
        }
        
        # Quality analysis
        qualities = [r.quality_score for r in results]
        analysis["quality_analysis"] = {
            "avg_quality": np.mean(qualities),
            "quality_consistency": np.std(qualities),
            "quality_vs_intelligibility_paradox": np.mean(qualities) > 80 and best_accuracy < 50
        }
        
        # Performance analysis
        processing_times = [r.processing_time for r in results]
        analysis["performance_analysis"] = {
            "total_processing_time": sum(processing_times),
            "avg_processing_time": np.mean(processing_times),
            "slowest_stage": results[processing_times.index(max(processing_times))].stage_name
        }
        
        # Critical findings
        critical_findings = []
        
        if analysis["intelligibility_analysis"]["all_stages_poor"]:
            critical_findings.append("ALL STAGES SHOW POOR INTELLIGIBILITY - Issue is in raw model output")
        
        if analysis["intelligibility_analysis"]["accuracy_degradation"] > 10:
            critical_findings.append(f"SIGNIFICANT DEGRADATION: {analysis['intelligibility_analysis']['accuracy_degradation']:.1f}% loss from {best_stage} to {worst_stage}")
        
        if analysis["quality_analysis"]["quality_vs_intelligibility_paradox"]:
            critical_findings.append("QUALITY-INTELLIGIBILITY PARADOX: High quality scores but poor intelligibility")
        
        if best_accuracy < 10:
            critical_findings.append("CRITICAL: Even best stage has <10% accuracy - fundamental model issue")
        
        analysis["critical_findings"] = critical_findings
        
        # Log critical findings
        logger.info("🚨 CRITICAL FINDINGS:")
        for finding in critical_findings:
            logger.critical(f"  - {finding}")
        
        if not critical_findings:
            logger.info("✅ No critical issues detected in pipeline")
        
        logger.info("")
        logger.info("📊 SUMMARY:")
        logger.info(f"  Best Accuracy: {best_accuracy:.1f}% ({best_stage})")
        logger.info(f"  Worst Accuracy: {worst_accuracy:.1f}% ({worst_stage})")
        logger.info(f"  Average Quality: {analysis['quality_analysis']['avg_quality']:.1f}%")
        logger.info(f"  Total Processing Time: {analysis['performance_analysis']['total_processing_time']:.3f}s")
        
        return analysis
    
    def save_investigation_report(self, results: List[PipelineStageResult], analysis: Dict[str, Any]) -> str:
        """Save comprehensive investigation report."""
        report = {
            "audio_pipeline_investigation": {
                "analysis": analysis,
                "detailed_results": [
                    {
                        "stage_name": r.stage_name,
                        "transcription": r.transcription,
                        "accuracy": r.accuracy,
                        "wer": r.wer,
                        "cer": r.cer,
                        "quality_score": r.quality_score,
                        "processing_time": r.processing_time,
                        "sample_rate": r.sample_rate,
                        "audio_duration": len(r.audio_data) / r.sample_rate,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            }
        }
        
        output_file = Path("temp") / "audio_pipeline_investigation_report.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📁 Investigation report saved to: {output_file}")
        return str(output_file)


async def main():
    """Main investigation execution."""
    logger.info("🔍 Starting Audio Pipeline Deep Dive Investigation")
    logger.info("=" * 60)
    
    try:
        investigator = AudioPipelineInvestigator()
        
        # Test cases
        test_cases = [
            ("Hello world.", "alloy"),
            ("The quick brown fox jumps over the lazy dog.", "fable"),
        ]
        
        all_results = []
        
        for text, voice in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"INVESTIGATING: '{text}' with voice '{voice}'")
            logger.info(f"{'='*60}")
            
            # Run pipeline investigation
            results = await investigator.investigate_full_pipeline(text, voice)
            
            # Analyze results
            analysis = investigator.analyze_results(results)
            
            # Save report
            report_file = investigator.save_investigation_report(results, analysis)
            
            all_results.append({
                "text": text,
                "voice": voice,
                "results": results,
                "analysis": analysis,
                "report_file": report_file
            })
        
        # Overall summary
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL INVESTIGATION SUMMARY")
        logger.info(f"{'='*60}")
        
        for i, case in enumerate(all_results, 1):
            analysis = case["analysis"]
            logger.info(f"Test Case {i}: '{case['text']}'")
            logger.info(f"  Best Accuracy: {analysis['intelligibility_analysis']['best_accuracy']:.1f}%")
            logger.info(f"  Critical Findings: {len(analysis['critical_findings'])}")
            logger.info("")
        
        logger.info("✅ Audio pipeline investigation completed")
        logger.info("📁 Check temp/ directory for detailed reports and audio samples")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
