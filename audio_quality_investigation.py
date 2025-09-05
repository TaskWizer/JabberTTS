#!/usr/bin/env python3
"""Audio Quality Investigation Tool for JabberTTS.

This tool performs comprehensive analysis of audio quality issues,
including raw model output validation, processing pipeline audit,
and format-specific quality analysis.
"""

import sys
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import io

# Try to import optional dependencies
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not available, using basic WAV writing")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available, using basic audio analysis")

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.models.speecht5 import SpeechT5Model
from jabbertts.config import get_settings


class AudioQualityInvestigator:
    """Comprehensive audio quality investigation tool."""
    
    def __init__(self):
        """Initialize the investigator."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.results = {}
        
        # Create output directory for analysis results
        self.output_dir = Path("audio_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Audio Quality Investigation Tool")
        print(f"Output directory: {self.output_dir}")
        print(f"Model: {self.settings.model_name}")
        print(f"Audio quality preset: {self.settings.audio_quality}")
        print()
    
    async def run_comprehensive_investigation(self) -> Dict[str, Any]:
        """Run comprehensive audio quality investigation."""
        print("=== Starting Comprehensive Audio Quality Investigation ===\n")
        
        # Phase 1: Raw Model Output Validation
        print("Phase 1: Raw Model Output Validation")
        raw_analysis = await self._analyze_raw_model_output()
        self.results["raw_model_analysis"] = raw_analysis
        
        # Phase 2: Audio Processing Pipeline Audit
        print("\nPhase 2: Audio Processing Pipeline Audit")
        pipeline_analysis = await self._audit_processing_pipeline()
        self.results["pipeline_analysis"] = pipeline_analysis
        
        # Phase 3: Format-Specific Quality Analysis
        print("\nPhase 3: Format-Specific Quality Analysis")
        format_analysis = await self._analyze_format_quality()
        self.results["format_analysis"] = format_analysis
        
        # Phase 4: Human-Perceptible Quality Assessment
        print("\nPhase 4: Human-Perceptible Quality Assessment")
        perceptual_analysis = await self._assess_perceptual_quality()
        self.results["perceptual_analysis"] = perceptual_analysis
        
        # Generate comprehensive report
        await self._generate_investigation_report()
        
        return self.results
    
    async def _analyze_raw_model_output(self) -> Dict[str, Any]:
        """Analyze raw model output before any processing."""
        print("  Analyzing raw SpeechT5 model output...")
        
        test_texts = [
            "Hello, world!",
            "This is a test of audio quality.",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        raw_analysis = {
            "model_info": {},
            "raw_samples": {},
            "spectral_analysis": {},
            "quality_metrics": {}
        }
        
        # Get model information
        if hasattr(self.engine, 'model') and self.engine.model:
            model = self.engine.model
            if hasattr(model, 'model'):
                raw_analysis["model_info"] = {
                    "model_type": type(model).__name__,
                    "sample_rate": getattr(model, 'SAMPLE_RATE', 'Unknown'),
                    "device": getattr(model, 'device', 'Unknown'),
                    "is_loaded": getattr(model, 'is_loaded', False)
                }
        
        for i, text in enumerate(test_texts):
            print(f"    Analyzing text {i+1}: '{text}'")
            
            try:
                # Generate raw audio from model
                result = await self.engine.generate_speech(text, voice="alloy")
                raw_audio = result["audio_data"]
                sample_rate = result["sample_rate"]
                
                # Save raw audio sample
                raw_filename = self.output_dir / f"raw_sample_{i+1}.wav"
                self._save_audio_file(raw_audio, sample_rate, raw_filename)

                # Analyze raw audio characteristics
                raw_metrics = self._analyze_audio_characteristics(raw_audio, sample_rate, f"raw_sample_{i+1}")

                # Generate basic analysis (spectrogram if available)
                analysis_info = self._generate_basic_analysis(raw_audio, sample_rate, f"raw_sample_{i+1}")
                
                raw_analysis["raw_samples"][f"sample_{i+1}"] = {
                    "text": text,
                    "audio_file": str(raw_filename),
                    "analysis_info": analysis_info,
                    "metrics": raw_metrics,
                    "sample_rate": sample_rate,
                    "duration": len(raw_audio) / sample_rate,
                    "shape": raw_audio.shape
                }

                print(f"      ✓ Raw audio saved: {raw_filename}")
                print(f"      ✓ Analysis completed: {analysis_info.get('status', 'basic')}")
                print(f"      ✓ Duration: {len(raw_audio) / sample_rate:.2f}s")
                print(f"      ✓ RMS: {np.sqrt(np.mean(raw_audio**2)):.4f}")
                print(f"      ✓ Peak: {np.max(np.abs(raw_audio)):.4f}")
                
            except Exception as e:
                print(f"      ✗ Error analyzing sample {i+1}: {e}")
                raw_analysis["raw_samples"][f"sample_{i+1}"] = {"error": str(e)}
        
        return raw_analysis
    
    async def _audit_processing_pipeline(self) -> Dict[str, Any]:
        """Audit the audio processing pipeline for quality issues."""
        print("  Auditing audio processing pipeline...")
        
        pipeline_analysis = {
            "processor_config": {},
            "processing_steps": {},
            "quality_degradation": {}
        }
        
        # Get processor configuration
        processor_info = self.audio_processor.get_processor_info()
        pipeline_analysis["processor_config"] = processor_info
        
        # Test processing pipeline with a sample
        test_text = "Testing audio processing pipeline quality."
        
        try:
            # Generate base audio
            result = await self.engine.generate_speech(test_text, voice="alloy")
            original_audio = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            print(f"    Original audio: {original_audio.shape} samples at {sample_rate}Hz")
            
            # Step 1: Normalization
            normalized_audio = self.audio_processor._normalize_audio(original_audio.copy())
            norm_metrics = self._analyze_audio_characteristics(normalized_audio, sample_rate, "normalized")
            
            # Step 2: Enhancement (if enabled)
            enhanced_audio = normalized_audio.copy()
            if self.settings.enable_audio_enhancement:
                enhanced_audio = self.audio_processor._enhance_audio(enhanced_audio, sample_rate)
            enh_metrics = self._analyze_audio_characteristics(enhanced_audio, sample_rate, "enhanced")
            
            # Step 3: Resampling (if needed)
            target_rate = self.audio_processor._get_target_sample_rate("mp3", sample_rate)
            resampled_audio = enhanced_audio
            if target_rate != sample_rate:
                resampled_audio, final_rate = self.audio_processor._resample_audio(enhanced_audio, sample_rate, target_rate)
            else:
                final_rate = sample_rate
            resamp_metrics = self._analyze_audio_characteristics(resampled_audio, final_rate, "resampled")
            
            pipeline_analysis["processing_steps"] = {
                "original": {
                    "metrics": self._analyze_audio_characteristics(original_audio, sample_rate, "original"),
                    "sample_rate": sample_rate,
                    "shape": original_audio.shape
                },
                "normalized": {
                    "metrics": norm_metrics,
                    "sample_rate": sample_rate,
                    "shape": normalized_audio.shape
                },
                "enhanced": {
                    "metrics": enh_metrics,
                    "sample_rate": sample_rate,
                    "shape": enhanced_audio.shape
                },
                "resampled": {
                    "metrics": resamp_metrics,
                    "sample_rate": final_rate,
                    "shape": resampled_audio.shape
                }
            }
            
            # Save intermediate samples for comparison
            self._save_audio_file(original_audio, sample_rate, self.output_dir / "pipeline_original.wav")
            self._save_audio_file(normalized_audio, sample_rate, self.output_dir / "pipeline_normalized.wav")
            self._save_audio_file(enhanced_audio, sample_rate, self.output_dir / "pipeline_enhanced.wav")
            self._save_audio_file(resampled_audio, final_rate, self.output_dir / "pipeline_resampled.wav")
            
            print(f"    ✓ Pipeline samples saved to {self.output_dir}")
            
            # Analyze quality degradation at each step
            pipeline_analysis["quality_degradation"] = self._analyze_quality_degradation(
                original_audio, normalized_audio, enhanced_audio, resampled_audio
            )
            
        except Exception as e:
            print(f"    ✗ Pipeline audit failed: {e}")
            pipeline_analysis["error"] = str(e)
        
        return pipeline_analysis
    
    async def _analyze_format_quality(self) -> Dict[str, Any]:
        """Analyze quality across different audio formats."""
        print("  Analyzing format-specific quality...")
        
        format_analysis = {
            "formats_tested": [],
            "format_comparison": {},
            "encoding_issues": {}
        }
        
        test_text = "Format quality comparison test sample."
        formats = ["wav", "mp3", "flac", "opus"]
        
        try:
            # Generate base audio
            result = await self.engine.generate_speech(test_text, voice="alloy")
            base_audio = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            format_analysis["formats_tested"] = formats
            
            for format_name in formats:
                print(f"    Testing format: {format_name}")
                
                try:
                    # Process audio to format
                    encoded_audio, metadata = await self.audio_processor.process_audio(
                        base_audio, sample_rate, output_format=format_name
                    )
                    
                    # Save encoded audio
                    format_filename = self.output_dir / f"format_test.{format_name}"
                    with open(format_filename, 'wb') as f:
                        f.write(encoded_audio)
                    
                    # Calculate compression metrics
                    original_size = len(base_audio) * 4  # float32 = 4 bytes per sample
                    compressed_size = len(encoded_audio)
                    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
                    
                    format_analysis["format_comparison"][format_name] = {
                        "file_path": str(format_filename),
                        "file_size": compressed_size,
                        "compression_ratio": compression_ratio,
                        "metadata": metadata,
                        "success": True
                    }
                    
                    print(f"      ✓ {format_name}: {compressed_size} bytes (ratio: {compression_ratio:.2f}x)")
                    
                except Exception as e:
                    print(f"      ✗ {format_name} failed: {e}")
                    format_analysis["format_comparison"][format_name] = {
                        "error": str(e),
                        "success": False
                    }
        
        except Exception as e:
            print(f"    ✗ Format analysis failed: {e}")
            format_analysis["error"] = str(e)
        
        return format_analysis
    
    async def _assess_perceptual_quality(self) -> Dict[str, Any]:
        """Assess human-perceptible quality issues."""
        print("  Assessing perceptual quality...")
        
        perceptual_analysis = {
            "intelligibility_tests": {},
            "artifact_detection": {},
            "naturalness_assessment": {}
        }
        
        # Test cases designed to reveal specific quality issues
        test_cases = [
            ("stuttering_test", "Testing for stuttering and robotic artifacts."),
            ("clarity_test", "Clear speech intelligibility validation."),
            ("naturalness_test", "Natural human-like voice quality assessment.")
        ]
        
        for test_name, test_text in test_cases:
            print(f"    Running {test_name}: '{test_text}'")
            
            try:
                # Generate audio
                result = await self.engine.generate_speech(test_text, voice="alloy")
                audio_data = result["audio_data"]
                sample_rate = result["sample_rate"]
                
                # Save test sample
                test_filename = self.output_dir / f"{test_name}.wav"
                self._save_audio_file(audio_data, sample_rate, test_filename)
                
                # Analyze for specific artifacts
                artifacts = self._detect_audio_artifacts(audio_data, sample_rate)
                
                perceptual_analysis["intelligibility_tests"][test_name] = {
                    "text": test_text,
                    "audio_file": str(test_filename),
                    "artifacts": artifacts,
                    "duration": len(audio_data) / sample_rate,
                    "quality_indicators": self._assess_quality_indicators(audio_data, sample_rate)
                }
                
                print(f"      ✓ {test_name} completed")
                
            except Exception as e:
                print(f"      ✗ {test_name} failed: {e}")
                perceptual_analysis["intelligibility_tests"][test_name] = {"error": str(e)}
        
        return perceptual_analysis
    
    def _analyze_audio_characteristics(self, audio: np.ndarray, sample_rate: int, label: str) -> Dict[str, Any]:
        """Analyze basic audio characteristics."""
        characteristics = {
            "rms_level": float(np.sqrt(np.mean(audio**2))),
            "peak_level": float(np.max(np.abs(audio))),
            "dynamic_range": float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-8))),
            "duration": float(len(audio) / sample_rate),
            "sample_count": int(len(audio))
        }

        if HAS_LIBROSA:
            try:
                characteristics.update({
                    "zero_crossing_rate": float(np.mean(librosa.zero_crossings(audio))),
                    "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))),
                    "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))),
                    "spectral_rolloff": float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)))
                })
            except Exception as e:
                characteristics["librosa_error"] = str(e)
        else:
            # Basic zero crossing rate calculation
            zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
            characteristics["zero_crossing_rate"] = float(len(zero_crossings) / len(audio))

        return characteristics
    
    def _generate_basic_analysis(self, audio: np.ndarray, sample_rate: int, label: str) -> Dict[str, Any]:
        """Generate basic audio analysis (without matplotlib)."""
        analysis = {
            "status": "basic",
            "label": label,
            "sample_rate": sample_rate,
            "duration": len(audio) / sample_rate
        }

        if HAS_LIBROSA:
            try:
                # Basic spectral analysis without plotting
                stft = librosa.stft(audio)
                magnitude = np.abs(stft)

                analysis.update({
                    "status": "librosa_available",
                    "spectral_shape": magnitude.shape,
                    "frequency_bins": magnitude.shape[0],
                    "time_frames": magnitude.shape[1],
                    "max_magnitude": float(np.max(magnitude)),
                    "mean_magnitude": float(np.mean(magnitude))
                })

                # Mel spectrogram analysis
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
                analysis["mel_spectrogram_shape"] = mel_spec.shape
                analysis["mel_energy"] = float(np.sum(mel_spec))

            except Exception as e:
                analysis["librosa_error"] = str(e)

        return analysis

    def _save_audio_file(self, audio: np.ndarray, sample_rate: int, filename: Path):
        """Save audio file using available libraries."""
        if HAS_SOUNDFILE:
            sf.write(filename, audio, sample_rate)
        else:
            # Basic WAV file writing
            self._write_basic_wav(audio, sample_rate, filename)

    def _write_basic_wav(self, audio: np.ndarray, sample_rate: int, filename: Path):
        """Write basic WAV file without soundfile."""
        import struct
        import wave

        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def _analyze_quality_degradation(self, original: np.ndarray, normalized: np.ndarray, 
                                   enhanced: np.ndarray, resampled: np.ndarray) -> Dict[str, Any]:
        """Analyze quality degradation at each processing step."""
        return {
            "normalization_impact": {
                "rms_change": float(np.sqrt(np.mean(normalized**2)) - np.sqrt(np.mean(original**2))),
                "peak_change": float(np.max(np.abs(normalized)) - np.max(np.abs(original))),
                "correlation": float(np.corrcoef(original.flatten(), normalized.flatten())[0, 1])
            },
            "enhancement_impact": {
                "rms_change": float(np.sqrt(np.mean(enhanced**2)) - np.sqrt(np.mean(normalized**2))),
                "peak_change": float(np.max(np.abs(enhanced)) - np.max(np.abs(normalized))),
                "correlation": float(np.corrcoef(normalized.flatten(), enhanced.flatten())[0, 1])
            },
            "resampling_impact": {
                "length_change": len(resampled) - len(enhanced),
                "rms_change": float(np.sqrt(np.mean(resampled**2)) - np.sqrt(np.mean(enhanced**2))),
                "peak_change": float(np.max(np.abs(resampled)) - np.max(np.abs(enhanced)))
            }
        }
    
    def _detect_audio_artifacts(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Detect specific audio artifacts."""
        artifacts = {}
        
        # Detect potential stuttering (rapid amplitude changes)
        amplitude_envelope = np.abs(audio)
        amplitude_diff = np.diff(amplitude_envelope)
        rapid_changes = np.sum(np.abs(amplitude_diff) > 0.1) / len(amplitude_diff)
        artifacts["stuttering_indicator"] = float(rapid_changes)
        
        # Detect silence gaps (potential robotic artifacts)
        silence_threshold = 0.01
        silence_mask = amplitude_envelope < silence_threshold
        silence_ratio = np.sum(silence_mask) / len(silence_mask)
        artifacts["silence_ratio"] = float(silence_ratio)
        
        # Detect frequency artifacts
        fft = np.fft.fft(audio)
        freq_magnitude = np.abs(fft)
        artifacts["frequency_peaks"] = int(np.sum(freq_magnitude > np.mean(freq_magnitude) * 10))
        
        return artifacts
    
    def _assess_quality_indicators(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Assess overall quality indicators."""
        indicators = {
            "snr_estimate": float(20 * np.log10(np.sqrt(np.mean(audio**2)) / (np.std(audio) + 1e-8))),
            "amplitude_variance": float(np.var(audio)),
            "amplitude_range": float(np.max(audio) - np.min(audio))
        }

        if HAS_LIBROSA:
            try:
                indicators.update({
                    "harmonic_ratio": float(np.mean(librosa.effects.harmonic(audio)) / (np.mean(np.abs(audio)) + 1e-8)),
                    "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(y=audio))),
                    "tempo_stability": float(np.std(librosa.beat.tempo(y=audio, sr=sample_rate)))
                })
            except Exception as e:
                indicators["librosa_error"] = str(e)
        else:
            # Basic frequency domain analysis
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft)
            indicators["frequency_peak_ratio"] = float(np.max(magnitude) / (np.mean(magnitude) + 1e-8))

        return indicators
    
    async def _generate_investigation_report(self):
        """Generate comprehensive investigation report."""
        print("\n  Generating comprehensive investigation report...")
        
        # Save detailed results as JSON
        results_file = self.output_dir / "audio_quality_investigation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = self.output_dir / "AUDIO_QUALITY_INVESTIGATION_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write("# Audio Quality Investigation Summary\n\n")
            f.write(f"**Investigation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model**: {self.settings.model_name}\n")
            f.write(f"**Audio Quality Preset**: {self.settings.audio_quality}\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Raw model analysis summary
            if "raw_model_analysis" in self.results:
                f.write("### Raw Model Output Analysis\n")
                raw_data = self.results["raw_model_analysis"]
                if "model_info" in raw_data:
                    f.write(f"- Model Type: {raw_data['model_info'].get('model_type', 'Unknown')}\n")
                    f.write(f"- Sample Rate: {raw_data['model_info'].get('sample_rate', 'Unknown')}\n")
                    f.write(f"- Device: {raw_data['model_info'].get('device', 'Unknown')}\n")
                f.write(f"- Raw samples generated: {len(raw_data.get('raw_samples', {}))}\n\n")
            
            # Pipeline analysis summary
            if "pipeline_analysis" in self.results:
                f.write("### Processing Pipeline Analysis\n")
                pipeline_data = self.results["pipeline_analysis"]
                if "processor_config" in pipeline_data:
                    config = pipeline_data["processor_config"]
                    f.write(f"- Supported formats: {config.get('supported_formats', [])}\n")
                    f.write(f"- Audio enhancement enabled: {config.get('audio_enhancement', {}).get('enabled', False)}\n")
                f.write("\n")
            
            # Format analysis summary
            if "format_analysis" in self.results:
                f.write("### Format Quality Analysis\n")
                format_data = self.results["format_analysis"]
                formats_tested = format_data.get("formats_tested", [])
                f.write(f"- Formats tested: {', '.join(formats_tested)}\n")
                
                if "format_comparison" in format_data:
                    for fmt, data in format_data["format_comparison"].items():
                        if data.get("success", False):
                            ratio = data.get("compression_ratio", 0)
                            f.write(f"- {fmt.upper()}: {data.get('file_size', 0)} bytes (compression: {ratio:.2f}x)\n")
                        else:
                            f.write(f"- {fmt.upper()}: FAILED - {data.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the investigation findings:\n\n")
            f.write("1. **Review raw model output quality** - Check spectrograms and waveforms in analysis results\n")
            f.write("2. **Analyze processing pipeline impact** - Compare intermediate processing steps\n")
            f.write("3. **Validate format encoding parameters** - Ensure optimal settings for each format\n")
            f.write("4. **Implement human listening tests** - Validate automated metrics with actual perception\n\n")
            
            f.write(f"## Files Generated\n\n")
            f.write(f"All analysis files are saved in: `{self.output_dir}/`\n\n")
            f.write("- Raw audio samples and spectrograms\n")
            f.write("- Processing pipeline intermediate files\n")
            f.write("- Format comparison samples\n")
            f.write("- Perceptual quality test samples\n")
            f.write("- Detailed JSON results\n")
        
        print(f"  ✓ Investigation complete! Results saved to {self.output_dir}")
        print(f"  ✓ Summary report: {summary_file}")
        print(f"  ✓ Detailed results: {results_file}")


async def main():
    """Run the audio quality investigation."""
    investigator = AudioQualityInvestigator()
    
    try:
        results = await investigator.run_comprehensive_investigation()
        print(f"\n=== Investigation Complete ===")
        print(f"Results saved to: {investigator.output_dir}")
        return results
    except Exception as e:
        print(f"\n✗ Investigation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
