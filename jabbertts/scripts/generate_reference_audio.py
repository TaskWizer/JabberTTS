#!/usr/bin/env python3
"""
Reference Audio Generation and Baseline Comparison Script.

This script generates standardized test samples with progressive processing stages
to isolate exact degradation sources and establish quality baselines.
"""

import asyncio
import logging
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.validation.whisper_validator import WhisperValidator
from jabbertts.validation.audio_quality import AudioQualityValidator
from jabbertts.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReferenceAudioGenerator:
    """Generate reference audio samples for baseline comparison."""
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.whisper_validator = WhisperValidator("base")
        self.quality_validator = AudioQualityValidator()
        self.settings = get_settings()
        
        # Create output directory
        self.output_dir = Path("reference_audio_samples")
        self.output_dir.mkdir(exist_ok=True)
        
        # Standard test phrases for different complexity levels
        self.test_phrases = {
            "simple": [
                "Hello world.",
                "This is a test.",
                "Good morning everyone.",
            ],
            "medium": [
                "The quick brown fox jumps over the lazy dog.",
                "Testing audio quality with various sentence structures.",
                "Natural speech synthesis requires careful attention to detail.",
            ],
            "complex": [
                "The sophisticated text-to-speech system demonstrates remarkable capabilities in generating human-like vocalizations.",
                "Comprehensive evaluation of audio quality involves multiple metrics including intelligibility, naturalness, and prosodic accuracy.",
                "Advanced neural networks have revolutionized the field of speech synthesis, enabling unprecedented levels of realism and expressiveness.",
            ]
        }
        
        # Standard voices to test
        self.test_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        # Processing stages for comparison
        self.processing_stages = [
            "raw_model",
            "minimal_processing", 
            "no_enhancement",
            "standard_processing",
            "high_quality_processing"
        ]
    
    async def generate_reference_sample(self, text: str, voice: str, stage: str) -> Dict[str, Any]:
        """Generate a single reference audio sample."""
        logger.info(f"Generating {stage} sample for voice '{voice}': '{text[:50]}...'")
        
        try:
            start_time = time.time()
            
            # Generate base TTS audio
            tts_result = await self.inference_engine.generate_speech(
                text=text,
                voice=voice,
                speed=1.0,
                response_format="wav"
            )
            
            audio_data = tts_result["audio_data"]
            sample_rate = tts_result["sample_rate"]
            
            # Apply processing based on stage
            if stage == "raw_model":
                # Save raw model output directly
                processed_audio = audio_data
                metadata = {
                    "processing": "raw_model_output",
                    "sample_rate": sample_rate,
                    "enhancement": False
                }
                
            elif stage == "minimal_processing":
                # Apply only basic normalization
                processed_audio = self._apply_minimal_processing(audio_data)
                metadata = {
                    "processing": "minimal_normalization_only",
                    "sample_rate": sample_rate,
                    "enhancement": False
                }
                
            elif stage == "no_enhancement":
                # Use AudioProcessor but disable enhancement
                original_enhancement = self.settings.enable_audio_enhancement
                self.settings.enable_audio_enhancement = False
                
                processed_bytes, proc_metadata = await self.audio_processor.process_audio(
                    audio_array=audio_data,
                    sample_rate=sample_rate,
                    output_format="wav",
                    speed=1.0
                )
                
                # Convert back to array for analysis
                import io
                with io.BytesIO(processed_bytes) as buffer:
                    processed_audio, _ = sf.read(buffer)
                
                self.settings.enable_audio_enhancement = original_enhancement
                metadata = proc_metadata
                metadata["processing"] = "no_enhancement"
                
            elif stage == "standard_processing":
                # Standard AudioProcessor settings
                processed_bytes, proc_metadata = await self.audio_processor.process_audio(
                    audio_array=audio_data,
                    sample_rate=sample_rate,
                    output_format="wav",
                    speed=1.0
                )
                
                # Convert back to array for analysis
                import io
                with io.BytesIO(processed_bytes) as buffer:
                    processed_audio, _ = sf.read(buffer)
                
                metadata = proc_metadata
                metadata["processing"] = "standard"
                
            elif stage == "high_quality_processing":
                # High quality settings
                original_quality = self.settings.audio_quality
                self.settings.audio_quality = "ultra"
                
                processed_bytes, proc_metadata = await self.audio_processor.process_audio(
                    audio_array=audio_data,
                    sample_rate=sample_rate,
                    output_format="wav",
                    speed=1.0
                )
                
                # Convert back to array for analysis
                import io
                with io.BytesIO(processed_bytes) as buffer:
                    processed_audio, _ = sf.read(buffer)
                
                self.settings.audio_quality = original_quality
                metadata = proc_metadata
                metadata["processing"] = "high_quality"
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            complexity = self._get_text_complexity(text)
            filename = f"ref_{stage}_{voice}_{complexity}_{timestamp}.wav"
            filepath = self.output_dir / filename
            
            # Save audio file
            sf.write(str(filepath), processed_audio, sample_rate)
            
            # Analyze quality
            quality_metrics = self.quality_validator.analyze_audio(processed_audio, sample_rate)
            
            # Whisper validation
            # Convert audio to bytes for Whisper
            import io
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, processed_audio, sample_rate, format='WAV')
            audio_bytes.seek(0)

            whisper_result = self.whisper_validator.validate_tts_output(
                original_text=text,
                audio_data=audio_bytes.getvalue(),
                sample_rate=sample_rate
            )
            
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "filepath": str(filepath),
                "text": text,
                "voice": voice,
                "stage": stage,
                "complexity": complexity,
                "generation_time": generation_time,
                "tts_metrics": {
                    "rtf": tts_result.get("rtf", 0),
                    "inference_time": tts_result.get("inference_time", 0),
                    "duration": tts_result.get("duration", 0)
                },
                "processing_metadata": metadata,
                "quality_metrics": {
                    "rms_level": float(quality_metrics.rms_level),
                    "peak_level": float(quality_metrics.peak_level),
                    "dynamic_range": float(quality_metrics.dynamic_range),
                    "spectral_centroid": float(quality_metrics.spectral_centroid),
                    "overall_quality": float(quality_metrics.overall_quality),
                    "naturalness_score": float(quality_metrics.naturalness_score),
                    "clarity_score": float(quality_metrics.clarity_score)
                },
                "whisper_validation": {
                    "transcription_accuracy": float(whisper_result.get("accuracy_score", 0)),
                    "transcribed_text": whisper_result.get("transcription", ""),
                    "character_accuracy": float(whisper_result.get("character_accuracy", 0)),
                    "word_accuracy": float(whisper_result.get("word_accuracy", 0))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate reference sample: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": text,
                "voice": voice,
                "stage": stage
            }
    
    def _apply_minimal_processing(self, audio: np.ndarray) -> np.ndarray:
        """Apply minimal processing (normalization only)."""
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio[:, 0] if audio.shape[1] > 0 else audio.flatten()
        
        # Basic normalization with conservative headroom
        max_val = np.abs(audio).max()
        if max_val > 0.85:
            audio = audio / max_val * 0.85
        
        return audio
    
    def _get_text_complexity(self, text: str) -> str:
        """Determine text complexity level."""
        if len(text) <= 20:
            return "simple"
        elif len(text) <= 100:
            return "medium"
        else:
            return "complex"
    
    async def generate_comprehensive_reference_set(self) -> Dict[str, Any]:
        """Generate comprehensive reference audio set."""
        logger.info("🎵 Starting Comprehensive Reference Audio Generation")
        logger.info("=" * 60)
        
        results = {
            "generation_timestamp": datetime.now().isoformat(),
            "samples": [],
            "summary": {},
            "baseline_metrics": {}
        }
        
        total_samples = 0
        successful_samples = 0
        
        # Generate samples for each combination
        for complexity, phrases in self.test_phrases.items():
            for phrase in phrases[:1]:  # Use first phrase of each complexity
                for voice in self.test_voices[:2]:  # Use first 2 voices for efficiency
                    for stage in self.processing_stages:
                        sample_result = await self.generate_reference_sample(phrase, voice, stage)
                        results["samples"].append(sample_result)
                        
                        total_samples += 1
                        if sample_result["success"]:
                            successful_samples += 1
        
        # Calculate summary statistics
        results["summary"] = {
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "success_rate": successful_samples / total_samples if total_samples > 0 else 0,
            "output_directory": str(self.output_dir)
        }
        
        # Save results to JSON
        results_file = self.output_dir / f"reference_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📊 Generation Complete:")
        logger.info(f"   Total samples: {total_samples}")
        logger.info(f"   Successful: {successful_samples}")
        logger.info(f"   Success rate: {results['summary']['success_rate']:.1%}")
        logger.info(f"   Results saved: {results_file}")
        
        return results

async def main():
    """Main execution function."""
    generator = ReferenceAudioGenerator()
    await generator.generate_comprehensive_reference_set()

if __name__ == "__main__":
    asyncio.run(main())
