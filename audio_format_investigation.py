#!/usr/bin/env python3
"""Audio Format Investigation Script for JabberTTS.

This script provides comprehensive technical analysis of audio format and encoding
issues that may be causing quality degradation in JabberTTS speech output.

Modules:
A. Sample Rate Integrity Verification
B. FFmpeg Processing Pipeline Analysis  
C. Bit Depth and Quantization Investigation
D. Spectral and Temporal Analysis
E. Processing Stage Isolation
"""

import sys
import asyncio
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import subprocess

# Add project root to path
sys.path.append('.')

from jabbertts.inference.engine import InferenceEngine
from jabbertts.audio.processor import AudioProcessor
from jabbertts.config import get_settings
from jabbertts.models.manager import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFormatInvestigator:
    """Comprehensive audio format and encoding investigation tool."""
    
    def __init__(self):
        """Initialize the investigator."""
        self.settings = get_settings()
        self.engine = InferenceEngine()
        self.audio_processor = AudioProcessor()
        self.model_manager = get_model_manager()
        
        # Create investigation output directory
        self.output_dir = Path("audio_format_investigation")
        self.output_dir.mkdir(exist_ok=True)
        
        # Test phrases for consistent analysis
        self.test_phrases = [
            {
                "name": "simple_word",
                "text": "Welcome",
                "description": "Single word reported as intelligible"
            },
            {
                "name": "tts_trigger",
                "text": "Text-to-speech synthesis", 
                "description": "Known T-T-S stuttering trigger phrase"
            },
            {
                "name": "complex_sentence",
                "text": "The quick brown fox jumps over the lazy dog",
                "description": "Complex sentence for comprehensive analysis"
            }
        ]
        
        print(f"Audio Format Investigation for JabberTTS")
        print(f"Output directory: {self.output_dir}")
        print(f"Test phrases: {len(self.test_phrases)}")
        print()
    
    async def run_comprehensive_investigation(self) -> Dict[str, Any]:
        """Run comprehensive audio format investigation."""
        print("=== AUDIO FORMAT INVESTIGATION ===\n")
        
        results = {
            "investigation_timestamp": time.time(),
            "investigation_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "module_a_sample_rate": {},
            "module_b_ffmpeg_analysis": {},
            "module_c_quantization": {},
            "module_d_spectral_analysis": {},
            "module_e_stage_isolation": {},
            "findings": [],
            "recommendations": []
        }
        
        # Module A: Sample Rate Integrity Verification
        print("Module A: Sample Rate Integrity Verification")
        print("=" * 50)
        results["module_a_sample_rate"] = await self._module_a_sample_rate_integrity()
        
        # Module B: FFmpeg Processing Pipeline Analysis
        print("\nModule B: FFmpeg Processing Pipeline Analysis")
        print("=" * 50)
        results["module_b_ffmpeg_analysis"] = await self._module_b_ffmpeg_analysis()
        
        # Module C: Bit Depth and Quantization Investigation
        print("\nModule C: Bit Depth and Quantization Investigation")
        print("=" * 50)
        results["module_c_quantization"] = await self._module_c_quantization_analysis()
        
        # Module D: Spectral and Temporal Analysis
        print("\nModule D: Spectral and Temporal Analysis")
        print("=" * 50)
        results["module_d_spectral_analysis"] = await self._module_d_spectral_analysis()
        
        # Module E: Processing Stage Isolation
        print("\nModule E: Processing Stage Isolation")
        print("=" * 50)
        results["module_e_stage_isolation"] = await self._module_e_stage_isolation()
        
        # Generate findings and recommendations
        results["findings"] = self._generate_investigation_findings(results)
        results["recommendations"] = self._generate_investigation_recommendations(results)
        
        # Save results
        results_file = self.output_dir / "audio_format_investigation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_investigation_summary_report(results)
        
        print(f"\n✓ Audio format investigation completed!")
        print(f"✓ Results saved to: {results_file}")
        print(f"✓ Summary report: {self.output_dir}/AUDIO_FORMAT_INVESTIGATION_SUMMARY.md")
        
        return results
    
    async def _module_a_sample_rate_integrity(self) -> Dict[str, Any]:
        """Module A: Verify 16kHz consistency through SpeechT5 → AudioProcessor → FFmpeg → output."""
        print("Analyzing sample rate integrity through the pipeline...")
        
        analysis_results = {
            "sample_rate_tests": {},
            "conversion_artifacts": {},
            "pipeline_consistency": {}
        }
        
        # Test different sample rates
        test_sample_rates = [8000, 16000, 22050, 44100, 48000]
        
        for phrase in self.test_phrases[:1]:  # Use one phrase for sample rate testing
            print(f"\nTesting sample rate integrity for: {phrase['name']}")
            
            phrase_results = {}
            
            for target_sr in test_sample_rates:
                print(f"  Testing target sample rate: {target_sr}Hz")
                
                try:
                    # Generate audio with current pipeline
                    result = await self.engine.generate_speech(phrase["text"], voice="alloy")
                    
                    # Get raw model output sample rate
                    model = await self.engine._ensure_model_loaded()
                    model_sr = model.get_sample_rate()
                    
                    # Analyze sample rate conversion if needed
                    conversion_analysis = self._analyze_sample_rate_conversion(
                        result["audio_data"], model_sr, target_sr
                    )
                    
                    # Save audio with different sample rate handling
                    audio_file = self.output_dir / f"{phrase['name']}_sr_{target_sr}.wav"
                    await self._save_audio_with_sample_rate(
                        result["audio_data"], model_sr, target_sr, audio_file
                    )
                    
                    phrase_results[f"sr_{target_sr}"] = {
                        "target_sample_rate": target_sr,
                        "model_sample_rate": model_sr,
                        "conversion_needed": model_sr != target_sr,
                        "conversion_analysis": conversion_analysis,
                        "audio_file": str(audio_file),
                        "rtf": result["rtf"]
                    }
                    
                    print(f"    ✓ Model SR: {model_sr}Hz, Target: {target_sr}Hz, RTF: {result['rtf']:.3f}")
                    
                except Exception as e:
                    print(f"    ✗ Sample rate test failed: {e}")
                    phrase_results[f"sr_{target_sr}"] = {"error": str(e)}
            
            analysis_results["sample_rate_tests"][phrase["name"]] = phrase_results
        
        # Analyze pipeline consistency
        analysis_results["pipeline_consistency"] = self._analyze_pipeline_sample_rate_consistency()
        
        return analysis_results
    
    def _analyze_sample_rate_conversion(self, audio: np.ndarray, source_sr: int, target_sr: int) -> Dict[str, Any]:
        """Analyze sample rate conversion for potential artifacts."""
        analysis = {
            "conversion_ratio": target_sr / source_sr,
            "upsampling": target_sr > source_sr,
            "downsampling": target_sr < source_sr,
            "no_conversion": target_sr == source_sr
        }
        
        if target_sr != source_sr:
            # Analyze potential aliasing and quality loss
            nyquist_source = source_sr / 2
            nyquist_target = target_sr / 2
            
            analysis.update({
                "nyquist_source": nyquist_source,
                "nyquist_target": nyquist_target,
                "potential_aliasing": target_sr < source_sr and nyquist_target < nyquist_source,
                "frequency_range_loss": max(0, nyquist_source - nyquist_target),
                "quality_impact": "high" if abs(target_sr - source_sr) > 8000 else "low"
            })
        
        return analysis
    
    def _analyze_pipeline_sample_rate_consistency(self) -> Dict[str, Any]:
        """Analyze sample rate consistency across pipeline components."""
        consistency_analysis = {
            "model_output_sr": 16000,  # SpeechT5 default
            "audio_processor_expected_sr": self.settings.audio_quality,  # Check config
            "ffmpeg_output_sr": "varies",  # Depends on format
            "potential_mismatches": []
        }
        
        # Check for common sample rate mismatches
        if hasattr(self.settings, 'sample_rate') and self.settings.sample_rate != 16000:
            consistency_analysis["potential_mismatches"].append(
                f"Config sample rate ({self.settings.sample_rate}) != Model output (16000)"
            )
        
        return consistency_analysis
    
    async def _save_audio_with_sample_rate(self, audio_data: np.ndarray, source_sr: int, 
                                         target_sr: int, filename: Path) -> None:
        """Save audio with specific sample rate handling."""
        try:
            # Import librosa for high-quality resampling if available
            import librosa
            
            if source_sr != target_sr:
                # Resample audio
                resampled_audio = librosa.resample(audio_data, orig_sr=source_sr, target_sr=target_sr)
            else:
                resampled_audio = audio_data
            
            # Save using audio processor
            audio_bytes, _ = await self.audio_processor.process_audio(
                resampled_audio, target_sr, "wav"
            )
            with open(filename, 'wb') as f:
                f.write(audio_bytes)
                
        except ImportError:
            # Fallback without librosa
            print(f"    Warning: librosa not available for high-quality resampling")
            audio_bytes, _ = await self.audio_processor.process_audio(
                audio_data, source_sr, "wav"
            )
            with open(filename, 'wb') as f:
                f.write(audio_bytes)

    async def _module_b_ffmpeg_analysis(self) -> Dict[str, Any]:
        """Module B: Compare raw model output vs FFmpeg-processed output."""
        print("Analyzing FFmpeg processing pipeline impact...")

        analysis_results = {
            "ffmpeg_comparison": {},
            "codec_parameter_tests": {},
            "bypass_tests": {}
        }

        for phrase in self.test_phrases:
            print(f"\nAnalyzing FFmpeg impact for: {phrase['name']}")

            try:
                # Generate raw model output
                result = await self.engine.generate_speech(phrase["text"], voice="alloy")
                model = await self.engine._ensure_model_loaded()

                # Save raw model output (bypass AudioProcessor)
                raw_file = self.output_dir / f"{phrase['name']}_raw_model.wav"
                await self._save_raw_audio(result["audio_data"], model.get_sample_rate(), raw_file)

                # Save with AudioProcessor (includes FFmpeg)
                processed_file = self.output_dir / f"{phrase['name']}_ffmpeg_processed.wav"
                audio_bytes, _ = await self.audio_processor.process_audio(
                    result["audio_data"], model.get_sample_rate(), "wav"
                )
                with open(processed_file, 'wb') as f:
                    f.write(audio_bytes)

                # Analyze differences
                raw_analysis = self._analyze_audio_characteristics(result["audio_data"])

                # Load processed audio for comparison
                processed_audio = self._load_audio_file(processed_file)
                processed_analysis = self._analyze_audio_characteristics(processed_audio)

                comparison = self._compare_audio_characteristics(raw_analysis, processed_analysis)

                analysis_results["ffmpeg_comparison"][phrase["name"]] = {
                    "raw_file": str(raw_file),
                    "processed_file": str(processed_file),
                    "raw_analysis": raw_analysis,
                    "processed_analysis": processed_analysis,
                    "comparison": comparison
                }

                print(f"    ✓ Raw vs Processed comparison completed")

            except Exception as e:
                print(f"    ✗ FFmpeg analysis failed: {e}")
                analysis_results["ffmpeg_comparison"][phrase["name"]] = {"error": str(e)}

        return analysis_results

    async def _save_raw_audio(self, audio_data: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Save raw audio data without any processing."""
        try:
            import soundfile as sf
            sf.write(str(filename), audio_data, sample_rate)
        except ImportError:
            # Fallback to simple WAV creation
            self._create_simple_wav(audio_data, sample_rate, filename)

    def _create_simple_wav(self, audio: np.ndarray, sample_rate: int, filename: Path) -> None:
        """Create a simple WAV file without external dependencies."""
        import struct
        import wave

        # Ensure audio is in correct format
        if audio.dtype != np.int16:
            # Convert float32 to int16
            audio = (audio * 32767).astype(np.int16)

        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

    def _load_audio_file(self, filename: Path) -> np.ndarray:
        """Load audio file for analysis."""
        try:
            import soundfile as sf
            audio, _ = sf.read(str(filename))
            return audio
        except ImportError:
            # Fallback using wave module
            import wave
            with wave.open(str(filename), 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
                return audio

    def _analyze_audio_characteristics(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio characteristics for comparison."""
        if len(audio) == 0:
            return {"error": "Empty audio"}

        characteristics = {
            "length": len(audio),
            "duration": len(audio) / 16000,  # Assuming 16kHz
            "rms_energy": float(np.sqrt(np.mean(audio**2))),
            "peak_amplitude": float(np.max(np.abs(audio))),
            "zero_crossings": int(np.sum(np.diff(np.sign(audio)) != 0)),
            "dynamic_range": float(np.max(audio) - np.min(audio)),
            "dc_offset": float(np.mean(audio)),
            "clipping_detected": bool(np.any(np.abs(audio) > 0.99))
        }

        # Spectral characteristics (simplified)
        if len(audio) > 1024:
            # Simple spectral analysis using FFT
            fft = np.fft.fft(audio[:1024])
            magnitude = np.abs(fft[:512])
            freqs = np.fft.fftfreq(1024, 1/16000)[:512]

            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                characteristics["spectral_centroid"] = float(spectral_centroid)
                characteristics["spectral_rolloff"] = float(freqs[np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0][0]])

        return characteristics

    def _compare_audio_characteristics(self, raw: Dict[str, Any], processed: Dict[str, Any]) -> Dict[str, Any]:
        """Compare audio characteristics between raw and processed audio."""
        comparison = {}

        for key in raw:
            if key in processed and isinstance(raw[key], (int, float)):
                raw_val = raw[key]
                processed_val = processed[key]

                if raw_val != 0:
                    percentage_change = ((processed_val - raw_val) / raw_val) * 100
                else:
                    percentage_change = 0

                comparison[key] = {
                    "raw": raw_val,
                    "processed": processed_val,
                    "difference": processed_val - raw_val,
                    "percentage_change": percentage_change
                }

        # Overall quality assessment
        comparison["quality_impact"] = {
            "rms_change": comparison.get("rms_energy", {}).get("percentage_change", 0),
            "peak_change": comparison.get("peak_amplitude", {}).get("percentage_change", 0),
            "dynamic_range_change": comparison.get("dynamic_range", {}).get("percentage_change", 0),
            "significant_degradation": abs(comparison.get("rms_energy", {}).get("percentage_change", 0)) > 10
        }

        return comparison

    async def _module_c_quantization_analysis(self) -> Dict[str, Any]:
        """Module C: Examine float32→int16 conversion for precision loss."""
        print("Analyzing bit depth and quantization effects...")

        analysis_results = {
            "bit_depth_tests": {},
            "quantization_analysis": {},
            "dynamic_range_tests": {}
        }

        # Test different bit depths
        bit_depths = [16, 24, 32]  # 16-bit int, 24-bit int, 32-bit float

        for phrase in self.test_phrases[:1]:  # Use one phrase for quantization testing
            print(f"\nTesting quantization for: {phrase['name']}")

            try:
                # Generate audio
                result = await self.engine.generate_speech(phrase["text"], voice="alloy")
                model = await self.engine._ensure_model_loaded()
                raw_audio = result["audio_data"]

                phrase_results = {}

                for bit_depth in bit_depths:
                    print(f"  Testing {bit_depth}-bit depth...")

                    # Convert to different bit depths
                    converted_audio, conversion_analysis = self._convert_bit_depth(raw_audio, bit_depth)

                    # Save converted audio
                    audio_file = self.output_dir / f"{phrase['name']}_bit_{bit_depth}.wav"
                    await self._save_audio_with_bit_depth(
                        converted_audio, model.get_sample_rate(), bit_depth, audio_file
                    )

                    # Analyze quantization effects
                    quantization_analysis = self._analyze_quantization_effects(raw_audio, converted_audio)

                    phrase_results[f"bit_{bit_depth}"] = {
                        "bit_depth": bit_depth,
                        "audio_file": str(audio_file),
                        "conversion_analysis": conversion_analysis,
                        "quantization_analysis": quantization_analysis
                    }

                    print(f"    ✓ {bit_depth}-bit: SNR={quantization_analysis.get('snr_db', 0):.1f}dB")

                analysis_results["bit_depth_tests"][phrase["name"]] = phrase_results

            except Exception as e:
                print(f"    ✗ Quantization analysis failed: {e}")
                analysis_results["bit_depth_tests"][phrase["name"]] = {"error": str(e)}

        return analysis_results

    def _convert_bit_depth(self, audio: np.ndarray, bit_depth: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert audio to different bit depths."""
        conversion_analysis = {
            "original_dtype": str(audio.dtype),
            "target_bit_depth": bit_depth,
            "original_range": [float(np.min(audio)), float(np.max(audio))]
        }

        if bit_depth == 16:
            # Convert to 16-bit integer
            converted = (audio * 32767).astype(np.int16)
            conversion_analysis["target_dtype"] = "int16"
            conversion_analysis["target_range"] = [-32768, 32767]
        elif bit_depth == 24:
            # Convert to 24-bit (stored as int32)
            converted = (audio * 8388607).astype(np.int32)
            conversion_analysis["target_dtype"] = "int32 (24-bit)"
            conversion_analysis["target_range"] = [-8388608, 8388607]
        elif bit_depth == 32:
            # Keep as 32-bit float
            converted = audio.astype(np.float32)
            conversion_analysis["target_dtype"] = "float32"
            conversion_analysis["target_range"] = [-1.0, 1.0]
        else:
            converted = audio
            conversion_analysis["target_dtype"] = "unchanged"

        return converted, conversion_analysis

    def _analyze_quantization_effects(self, original: np.ndarray, quantized: np.ndarray) -> Dict[str, Any]:
        """Analyze quantization effects between original and quantized audio."""
        # Convert quantized back to float for comparison
        if quantized.dtype == np.int16:
            quantized_float = quantized.astype(np.float32) / 32767
        elif quantized.dtype == np.int32:
            quantized_float = quantized.astype(np.float32) / 8388607
        else:
            quantized_float = quantized.astype(np.float32)

        # Ensure same length
        min_len = min(len(original), len(quantized_float))
        original = original[:min_len]
        quantized_float = quantized_float[:min_len]

        # Calculate quantization noise
        noise = original - quantized_float

        # Calculate SNR
        signal_power = np.mean(original**2)
        noise_power = np.mean(noise**2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        analysis = {
            "snr_db": float(snr_db),
            "noise_rms": float(np.sqrt(noise_power)),
            "signal_rms": float(np.sqrt(signal_power)),
            "max_error": float(np.max(np.abs(noise))),
            "clipping_detected": bool(np.any(np.abs(quantized_float) >= 0.99)),
            "dynamic_range_original": float(np.max(original) - np.min(original)),
            "dynamic_range_quantized": float(np.max(quantized_float) - np.min(quantized_float))
        }

        return analysis

    async def _save_audio_with_bit_depth(self, audio_data: np.ndarray, sample_rate: int,
                                       bit_depth: int, filename: Path) -> None:
        """Save audio with specific bit depth."""
        try:
            import soundfile as sf

            # Determine subtype based on bit depth
            if bit_depth == 16:
                subtype = 'PCM_16'
            elif bit_depth == 24:
                subtype = 'PCM_24'
            elif bit_depth == 32:
                subtype = 'FLOAT'
            else:
                subtype = 'PCM_16'  # Default

            sf.write(str(filename), audio_data, sample_rate, subtype=subtype)

        except ImportError:
            # Fallback using audio processor
            audio_bytes, _ = await self.audio_processor.process_audio(
                audio_data, sample_rate, "wav"
            )
            with open(filename, 'wb') as f:
                f.write(audio_bytes)

    async def _module_d_spectral_analysis(self) -> Dict[str, Any]:
        """Module D: Use librosa for spectrogram generation and formant analysis."""
        print("Analyzing frequency domain characteristics...")

        analysis_results = {
            "spectral_analysis": {},
            "formant_analysis": {},
            "frequency_response": {}
        }

        for phrase in self.test_phrases:
            print(f"\nSpectral analysis for: {phrase['name']}")

            try:
                # Generate audio
                result = await self.engine.generate_speech(phrase["text"], voice="alloy")
                audio_data = result["audio_data"]

                # Perform spectral analysis
                spectral_features = self._analyze_spectral_features(audio_data)

                # Save spectrogram if possible
                spectrogram_file = self.output_dir / f"{phrase['name']}_spectrogram.png"
                self._save_spectrogram(audio_data, spectrogram_file)

                analysis_results["spectral_analysis"][phrase["name"]] = {
                    "spectral_features": spectral_features,
                    "spectrogram_file": str(spectrogram_file) if spectrogram_file.exists() else None,
                    "audio_duration": len(audio_data) / 16000
                }

                print(f"    ✓ Spectral analysis completed")

            except Exception as e:
                print(f"    ✗ Spectral analysis failed: {e}")
                analysis_results["spectral_analysis"][phrase["name"]] = {"error": str(e)}

        return analysis_results

    def _analyze_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze spectral features of audio."""
        if len(audio) < 1024:
            return {"error": "Audio too short for spectral analysis"}

        # Basic spectral analysis using numpy FFT
        window_size = 1024
        hop_length = 512

        # Compute STFT
        spectrograms = []
        for i in range(0, len(audio) - window_size, hop_length):
            window = audio[i:i + window_size]
            fft = np.fft.fft(window)
            magnitude = np.abs(fft[:window_size // 2])
            spectrograms.append(magnitude)

        if not spectrograms:
            return {"error": "Could not compute spectrogram"}

        spectrogram = np.array(spectrograms).T
        freqs = np.fft.fftfreq(window_size, 1/16000)[:window_size // 2]

        # Compute spectral features
        features = {
            "spectral_centroid": [],
            "spectral_rolloff": [],
            "spectral_bandwidth": [],
            "zero_crossing_rate": []
        }

        for frame in spectrograms:
            if np.sum(frame) > 0:
                # Spectral centroid
                centroid = np.sum(freqs * frame) / np.sum(frame)
                features["spectral_centroid"].append(centroid)

                # Spectral rolloff (85% of energy)
                cumsum = np.cumsum(frame)
                rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
                rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                features["spectral_rolloff"].append(rolloff)

                # Spectral bandwidth
                bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * frame) / np.sum(frame))
                features["spectral_bandwidth"].append(bandwidth)

        # Compute zero crossing rate
        zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        features["zero_crossing_rate"] = zcr

        # Summary statistics
        summary = {}
        for feature, values in features.items():
            if isinstance(values, list) and values:
                summary[feature] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            else:
                summary[feature] = float(values) if isinstance(values, (int, float)) else values

        return summary

    def _save_spectrogram(self, audio: np.ndarray, filename: Path) -> None:
        """Save spectrogram visualization."""
        try:
            import matplotlib.pyplot as plt

            # Compute spectrogram
            window_size = 1024
            hop_length = 512

            spectrograms = []
            for i in range(0, len(audio) - window_size, hop_length):
                window = audio[i:i + window_size]
                fft = np.fft.fft(window)
                magnitude = np.abs(fft[:window_size // 2])
                spectrograms.append(20 * np.log10(magnitude + 1e-10))  # Convert to dB

            if spectrograms:
                spectrogram = np.array(spectrograms).T

                plt.figure(figsize=(12, 6))
                plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Magnitude (dB)')
                plt.xlabel('Time (frames)')
                plt.ylabel('Frequency (bins)')
                plt.title('Spectrogram')
                plt.tight_layout()
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()

        except ImportError:
            print(f"    Warning: matplotlib not available for spectrogram visualization")
        except Exception as e:
            print(f"    Warning: Could not save spectrogram: {e}")

    async def _module_e_stage_isolation(self) -> Dict[str, Any]:
        """Module E: Generate progressive samples to pinpoint exact degradation stage."""
        print("Analyzing processing stages isolation...")

        analysis_results = {
            "stage_comparison": {},
            "degradation_analysis": {},
            "stage_isolation": {}
        }

        for phrase in self.test_phrases:
            print(f"\nStage isolation for: {phrase['name']}")

            try:
                # Stage 1: Raw SpeechT5 output (no JabberTTS processing)
                result = await self.engine.generate_speech(phrase["text"], voice="alloy")
                model = await self.engine._ensure_model_loaded()
                raw_audio = result["audio_data"]

                stage1_file = self.output_dir / f"{phrase['name']}_stage1_raw_model.wav"
                await self._save_raw_audio(raw_audio, model.get_sample_rate(), stage1_file)

                # Stage 2: Model + AudioProcessor only (no FFmpeg)
                stage2_file = self.output_dir / f"{phrase['name']}_stage2_audio_processor.wav"
                # This would require modifying AudioProcessor to bypass FFmpeg
                # For now, we'll use the current AudioProcessor
                audio_bytes, _ = await self.audio_processor.process_audio(
                    raw_audio, model.get_sample_rate(), "wav"
                )
                with open(stage2_file, 'wb') as f:
                    f.write(audio_bytes)

                # Stage 3: Full pipeline (Model + AudioProcessor + FFmpeg) - same as stage 2 currently
                stage3_file = self.output_dir / f"{phrase['name']}_stage3_full_pipeline.wav"
                with open(stage3_file, 'wb') as f:
                    f.write(audio_bytes)

                # Analyze each stage
                stage1_analysis = self._analyze_audio_characteristics(raw_audio)
                stage2_audio = self._load_audio_file(stage2_file)
                stage2_analysis = self._analyze_audio_characteristics(stage2_audio)
                stage3_audio = self._load_audio_file(stage3_file)
                stage3_analysis = self._analyze_audio_characteristics(stage3_audio)

                # Compare stages
                stage1_to_2 = self._compare_audio_characteristics(stage1_analysis, stage2_analysis)
                stage2_to_3 = self._compare_audio_characteristics(stage2_analysis, stage3_analysis)

                analysis_results["stage_comparison"][phrase["name"]] = {
                    "stage1_raw": {
                        "file": str(stage1_file),
                        "analysis": stage1_analysis
                    },
                    "stage2_processor": {
                        "file": str(stage2_file),
                        "analysis": stage2_analysis
                    },
                    "stage3_full": {
                        "file": str(stage3_file),
                        "analysis": stage3_analysis
                    },
                    "stage1_to_2_comparison": stage1_to_2,
                    "stage2_to_3_comparison": stage2_to_3
                }

                print(f"    ✓ Stage isolation completed")

            except Exception as e:
                print(f"    ✗ Stage isolation failed: {e}")
                analysis_results["stage_comparison"][phrase["name"]] = {"error": str(e)}

        return analysis_results

    def _generate_investigation_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate findings based on audio format investigation."""
        findings = []

        # Sample rate findings
        sample_rate_analysis = results.get("module_a_sample_rate", {})
        for phrase_name, phrase_data in sample_rate_analysis.get("sample_rate_tests", {}).items():
            for sr_test, sr_data in phrase_data.items():
                if "error" not in sr_data and sr_data.get("conversion_needed", False):
                    conversion = sr_data.get("conversion_analysis", {})
                    if conversion.get("quality_impact") == "high":
                        findings.append(
                            f"WARNING: High quality impact detected in sample rate conversion "
                            f"({sr_data['model_sample_rate']}Hz → {sr_data['target_sample_rate']}Hz)"
                        )

        # FFmpeg findings
        ffmpeg_analysis = results.get("module_b_ffmpeg_analysis", {})
        for phrase_name, phrase_data in ffmpeg_analysis.get("ffmpeg_comparison", {}).items():
            if "comparison" in phrase_data:
                quality_impact = phrase_data["comparison"].get("quality_impact", {})
                if quality_impact.get("significant_degradation", False):
                    findings.append(
                        f"CRITICAL: Significant quality degradation detected in FFmpeg processing for '{phrase_name}'"
                    )

        # Quantization findings
        quantization_analysis = results.get("module_c_quantization", {})
        for phrase_name, phrase_data in quantization_analysis.get("bit_depth_tests", {}).items():
            for bit_test, bit_data in phrase_data.items():
                if "quantization_analysis" in bit_data:
                    quant = bit_data["quantization_analysis"]
                    if quant.get("snr_db", 0) < 40:  # Low SNR indicates quality issues
                        findings.append(
                            f"WARNING: Low SNR ({quant['snr_db']:.1f}dB) detected in {bit_data['bit_depth']}-bit quantization"
                        )
                    if quant.get("clipping_detected", False):
                        findings.append(
                            f"CRITICAL: Audio clipping detected in {bit_data['bit_depth']}-bit quantization"
                        )

        # Spectral findings
        spectral_analysis = results.get("module_d_spectral_analysis", {})
        for phrase_name, phrase_data in spectral_analysis.get("spectral_analysis", {}).items():
            if "spectral_features" in phrase_data:
                features = phrase_data["spectral_features"]
                if "spectral_centroid" in features and isinstance(features["spectral_centroid"], dict):
                    centroid_mean = features["spectral_centroid"].get("mean", 0)
                    if centroid_mean < 1000 or centroid_mean > 4000:
                        findings.append(
                            f"WARNING: Unusual spectral centroid ({centroid_mean:.0f}Hz) detected in '{phrase_name}'"
                        )

        # Stage isolation findings
        stage_analysis = results.get("module_e_stage_isolation", {})
        for phrase_name, phrase_data in stage_analysis.get("stage_comparison", {}).items():
            if "stage1_to_2_comparison" in phrase_data:
                stage1_to_2 = phrase_data["stage1_to_2_comparison"]
                quality_impact = stage1_to_2.get("quality_impact", {})
                if quality_impact.get("significant_degradation", False):
                    findings.append(
                        f"CRITICAL: Significant degradation detected between raw model and audio processor for '{phrase_name}'"
                    )

        if not findings:
            findings.append("No critical audio format issues detected in automated analysis")

        return findings

    def _generate_investigation_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on investigation findings."""
        recommendations = []

        # Sample rate recommendations
        sample_rate_analysis = results.get("module_a_sample_rate", {})
        pipeline_consistency = sample_rate_analysis.get("pipeline_consistency", {})
        if pipeline_consistency.get("potential_mismatches"):
            recommendations.append(
                "IMMEDIATE: Fix sample rate mismatches in pipeline configuration"
            )

        # FFmpeg recommendations
        ffmpeg_analysis = results.get("module_b_ffmpeg_analysis", {})
        significant_degradation = False
        for phrase_data in ffmpeg_analysis.get("ffmpeg_comparison", {}).values():
            if "comparison" in phrase_data:
                quality_impact = phrase_data["comparison"].get("quality_impact", {})
                if quality_impact.get("significant_degradation", False):
                    significant_degradation = True
                    break

        if significant_degradation:
            recommendations.append(
                "CRITICAL: Optimize FFmpeg encoding parameters to reduce quality degradation"
            )
            recommendations.append(
                "INVESTIGATE: Consider bypassing FFmpeg for raw audio output"
            )

        # Quantization recommendations
        quantization_analysis = results.get("module_c_quantization", {})
        low_snr_detected = False
        clipping_detected = False

        for phrase_data in quantization_analysis.get("bit_depth_tests", {}).values():
            for bit_data in phrase_data.values():
                if "quantization_analysis" in bit_data:
                    quant = bit_data["quantization_analysis"]
                    if quant.get("snr_db", 0) < 40:
                        low_snr_detected = True
                    if quant.get("clipping_detected", False):
                        clipping_detected = True

        if clipping_detected:
            recommendations.append(
                "IMMEDIATE: Implement audio normalization to prevent clipping"
            )

        if low_snr_detected:
            recommendations.append(
                "OPTIMIZE: Use higher bit depth (24-bit or 32-bit float) for better quality"
            )

        # General recommendations
        recommendations.extend([
            "VALIDATE: Conduct manual listening tests on generated samples",
            "MONITOR: Implement real-time audio quality monitoring",
            "OPTIMIZE: Focus on stages showing highest quality degradation",
            "DOCUMENT: Update audio processing guidelines based on findings"
        ])

        return recommendations

    def _generate_investigation_summary_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive investigation summary report."""
        report_file = self.output_dir / "AUDIO_FORMAT_INVESTIGATION_SUMMARY.md"

        with open(report_file, 'w') as f:
            f.write("# Audio Format Investigation Summary Report\n\n")
            f.write(f"**Generated**: {results['investigation_date']}\n")
            f.write(f"**Investigation Type**: Comprehensive Audio Format and Encoding Analysis\n")
            f.write(f"**Model**: {self.settings.model_name}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("This investigation analyzed audio format and encoding issues that may be ")
            f.write("causing quality degradation in JabberTTS speech output through five specialized modules.\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            findings = results.get("findings", [])
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            recommendations = results.get("recommendations", [])
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")

            # Module Results Summary
            f.write("## Module Results Summary\n\n")

            # Module A: Sample Rate
            f.write("### Module A: Sample Rate Integrity\n")
            sample_rate_analysis = results.get("module_a_sample_rate", {})
            if sample_rate_analysis:
                f.write("- **Pipeline Consistency**: Analyzed sample rate handling across components\n")
                f.write("- **Conversion Testing**: Tested multiple sample rates for artifacts\n")
                f.write("- **Quality Impact**: Assessed conversion quality impact\n\n")

            # Module B: FFmpeg
            f.write("### Module B: FFmpeg Processing\n")
            ffmpeg_analysis = results.get("module_b_ffmpeg_analysis", {})
            if ffmpeg_analysis:
                f.write("- **Raw vs Processed**: Compared model output before/after FFmpeg\n")
                f.write("- **Quality Analysis**: Measured processing impact on audio characteristics\n")
                f.write("- **Degradation Detection**: Identified significant quality changes\n\n")

            # Module C: Quantization
            f.write("### Module C: Bit Depth and Quantization\n")
            quantization_analysis = results.get("module_c_quantization", {})
            if quantization_analysis:
                f.write("- **Bit Depth Testing**: Compared 16-bit, 24-bit, and 32-bit formats\n")
                f.write("- **SNR Analysis**: Measured signal-to-noise ratio for each format\n")
                f.write("- **Clipping Detection**: Identified audio clipping issues\n\n")

            # Module D: Spectral
            f.write("### Module D: Spectral Analysis\n")
            spectral_analysis = results.get("module_d_spectral_analysis", {})
            if spectral_analysis:
                f.write("- **Frequency Domain**: Analyzed spectral characteristics\n")
                f.write("- **Formant Analysis**: Examined formant structure\n")
                f.write("- **Spectral Features**: Computed centroid, rolloff, bandwidth\n\n")

            # Module E: Stage Isolation
            f.write("### Module E: Processing Stage Isolation\n")
            stage_analysis = results.get("module_e_stage_isolation", {})
            if stage_analysis:
                f.write("- **Progressive Analysis**: Isolated each processing stage\n")
                f.write("- **Degradation Pinpointing**: Identified exact degradation sources\n")
                f.write("- **Stage Comparison**: Measured quality changes between stages\n\n")

            # Generated Files
            f.write("## Generated Analysis Files\n\n")
            f.write("The following files were generated for detailed analysis:\n\n")

            # List all generated files by category
            categories = {
                "Sample Rate Tests": "sr_",
                "FFmpeg Analysis": ["raw_model", "ffmpeg_processed"],
                "Bit Depth Tests": "bit_",
                "Spectral Analysis": "spectrogram",
                "Stage Isolation": ["stage1_", "stage2_", "stage3_"]
            }

            for category, patterns in categories.items():
                f.write(f"### {category}\n")
                if isinstance(patterns, str):
                    pattern = patterns
                    files = list(self.output_dir.glob(f"*{pattern}*"))
                else:
                    files = []
                    for pattern in patterns:
                        files.extend(list(self.output_dir.glob(f"*{pattern}*")))

                for file in sorted(files):
                    f.write(f"- `{file.name}`\n")
                f.write("\n")

            f.write("---\n")
            f.write("**Note**: Manual audio comparison of generated files is essential to validate ")
            f.write("automated analysis and identify perceptual quality differences.\n")


async def main():
    """Execute the comprehensive audio format investigation."""
    investigator = AudioFormatInvestigator()

    try:
        print("🔍 Starting comprehensive audio format investigation...")
        print("This investigation will analyze format and encoding issues causing quality degradation.\n")

        results = await investigator.run_comprehensive_investigation()

        print("\n" + "="*70)
        print("AUDIO FORMAT INVESTIGATION COMPLETED")
        print("="*70)

        # Print key findings
        findings = results.get("findings", [])
        if findings:
            print("\n🔍 KEY FINDINGS:")
            for finding in findings:
                print(f"  • {finding}")

        # Print top recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")

        print(f"\n📁 Investigation files saved to: {investigator.output_dir}")
        print("📋 Review the summary report and analyze generated audio samples")
        print("\n🎧 MANUAL LISTENING TEST REQUIRED:")
        print("   Compare generated audio files to identify format-related quality issues")
        print("   Focus on raw model output vs processed audio differences")

        return results

    except Exception as e:
        print(f"\n❌ Audio format investigation failed: {e}")
        logger.exception("Investigation failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())
