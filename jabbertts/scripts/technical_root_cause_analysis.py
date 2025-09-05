#!/usr/bin/env python3
"""
Technical Root Cause Analysis Script

This script performs deep technical analysis of the SpeechT5 implementation to identify
the exact cause of intelligibility issues. It examines:
1. Text preprocessing pipeline step-by-step
2. Model input/output tensors
3. Speaker embeddings validation
4. Memory allocation patterns
5. Sequence length limitations
6. Audio generation pipeline

Usage:
    python jabbertts/scripts/technical_root_cause_analysis.py
"""

import asyncio
import json
import logging
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import gc

import numpy as np
import torch
import soundfile as sf

from jabbertts.models.manager import get_model_manager
from jabbertts.inference.preprocessing import TextPreprocessor
from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Deep technical analysis of TTS pipeline."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.model_manager = None
        self.model = None
        self.preprocessor = None
        self.whisper_validator = None
        self.analysis_results = {}
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("🔧 Initializing Technical Analysis Components")
        
        self.model_manager = get_model_manager()
        self.preprocessor = TextPreprocessor(use_phonemizer=True)
        self.whisper_validator = get_whisper_validator("base")
        
        # Load SpeechT5 model
        self.model = self.model_manager.load_model("speecht5")
        
        logger.info("✅ All components initialized")
    
    def analyze_text_preprocessing(self, text: str) -> Dict[str, Any]:
        """Analyze text preprocessing pipeline step by step."""
        logger.info(f"🔍 Analyzing text preprocessing for: '{text}'")
        
        results = {
            "original_text": text,
            "preprocessing_steps": {}
        }
        
        # Step 1: Basic cleaning
        cleaned = self.preprocessor._basic_cleaning(text)
        results["preprocessing_steps"]["1_basic_cleaning"] = cleaned
        logger.debug(f"Basic cleaning: '{text}' → '{cleaned}'")
        
        # Step 2: Unicode normalization
        normalized = self.preprocessor._normalize_unicode(cleaned)
        results["preprocessing_steps"]["2_unicode_normalization"] = normalized
        logger.debug(f"Unicode normalization: '{cleaned}' → '{normalized}'")
        
        # Step 3: Abbreviation expansion
        expanded = self.preprocessor._expand_abbreviations(normalized)
        results["preprocessing_steps"]["3_abbreviation_expansion"] = expanded
        logger.debug(f"Abbreviation expansion: '{normalized}' → '{expanded}'")
        
        # Step 4: Number normalization
        numbers_normalized = self.preprocessor._normalize_numbers(expanded)
        results["preprocessing_steps"]["4_number_normalization"] = numbers_normalized
        logger.debug(f"Number normalization: '{expanded}' → '{numbers_normalized}'")
        
        # Step 5: Punctuation handling
        punctuation_handled = self.preprocessor._handle_punctuation(numbers_normalized)
        results["preprocessing_steps"]["5_punctuation_handling"] = punctuation_handled
        logger.debug(f"Punctuation handling: '{numbers_normalized}' → '{punctuation_handled}'")
        
        # Step 6: Phonemization (if enabled)
        if self.preprocessor.use_phonemizer and self.preprocessor.phonemizer:
            try:
                phonemized = self.preprocessor._phonemize(punctuation_handled, "en")
                results["preprocessing_steps"]["6_phonemization"] = phonemized
                logger.debug(f"Phonemization: '{punctuation_handled}' → '{phonemized}'")
            except Exception as e:
                results["preprocessing_steps"]["6_phonemization"] = f"ERROR: {e}"
                logger.error(f"Phonemization failed: {e}")
        else:
            results["preprocessing_steps"]["6_phonemization"] = "DISABLED"
            logger.debug("Phonemization disabled")
        
        # Final result
        final_text = self.preprocessor.preprocess(text)
        results["final_preprocessed_text"] = final_text
        results["length_change"] = len(final_text) - len(text)
        
        logger.info(f"Preprocessing complete: '{text}' → '{final_text}'")
        return results
    
    def analyze_model_inputs(self, text: str) -> Dict[str, Any]:
        """Analyze model input tensors and tokenization."""
        logger.info(f"🔍 Analyzing model inputs for: '{text}'")
        
        results = {}
        
        try:
            # Process text through SpeechT5 processor
            inputs = self.model.processor(text=text, return_tensors="pt")
            
            results["input_ids"] = {
                "shape": list(inputs["input_ids"].shape),
                "dtype": str(inputs["input_ids"].dtype),
                "min_value": int(inputs["input_ids"].min()),
                "max_value": int(inputs["input_ids"].max()),
                "sequence_length": inputs["input_ids"].shape[1],
                "first_10_tokens": inputs["input_ids"][0][:10].tolist(),
                "last_10_tokens": inputs["input_ids"][0][-10:].tolist()
            }
            
            # Decode tokens back to text to verify tokenization
            decoded_text = self.model.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            results["decoded_text"] = decoded_text
            results["tokenization_roundtrip_match"] = (decoded_text.strip().lower() == text.strip().lower())
            
            # Check for special tokens
            special_tokens = []
            for token_id in inputs["input_ids"][0]:
                token = self.model.processor.tokenizer.decode([token_id])
                if token in self.model.processor.tokenizer.special_tokens_map.values():
                    special_tokens.append(token)
            results["special_tokens"] = special_tokens
            
            logger.info(f"Input analysis complete - sequence length: {results['input_ids']['sequence_length']}")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Model input analysis failed: {e}")
            traceback.print_exc()
        
        return results
    
    def analyze_speaker_embeddings(self, voice: str) -> Dict[str, Any]:
        """Analyze speaker embeddings for the given voice."""
        logger.info(f"🔍 Analyzing speaker embeddings for voice: '{voice}'")
        
        results = {}
        
        try:
            embeddings = self.model._get_speaker_embeddings(voice)
            
            results["shape"] = list(embeddings.shape)
            results["dtype"] = str(embeddings.dtype)
            results["mean"] = float(embeddings.mean())
            results["std"] = float(embeddings.std())
            results["min"] = float(embeddings.min())
            results["max"] = float(embeddings.max())
            results["norm"] = float(torch.norm(embeddings))
            results["has_nan"] = bool(torch.isnan(embeddings).any())
            results["has_inf"] = bool(torch.isinf(embeddings).any())
            
            # Check if embeddings are normalized
            norm_per_sample = torch.norm(embeddings, dim=1)
            results["is_normalized"] = bool(torch.allclose(norm_per_sample, torch.ones_like(norm_per_sample), atol=1e-3))
            
            logger.info(f"Speaker embeddings analysis complete - shape: {results['shape']}, norm: {results['norm']:.3f}")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Speaker embeddings analysis failed: {e}")
            traceback.print_exc()
        
        return results
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        logger.info("🔍 Analyzing memory usage")
        
        results = {}
        
        try:
            # Python memory
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            results["python_memory"] = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024
            }
            
            # PyTorch memory
            if torch.cuda.is_available():
                results["cuda_memory"] = {
                    "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
                }
            else:
                results["cuda_memory"] = "Not available (CPU only)"
            
            # Model memory usage
            if self.model:
                model_memory = self.model.get_memory_usage()
                results["model_memory"] = model_memory
            
            logger.info(f"Memory analysis complete - Python RSS: {results['python_memory']['rss_mb']:.1f}MB")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Memory analysis failed: {e}")
        
        return results
    
    def analyze_audio_generation(self, text: str, voice: str = "alloy") -> Dict[str, Any]:
        """Analyze the audio generation process step by step."""
        logger.info(f"🔍 Analyzing audio generation for: '{text}' with voice '{voice}'")
        
        results = {}
        
        try:
            # Step 1: Preprocess text
            processed_text = text  # SpeechT5 should use raw text
            results["processed_text"] = processed_text
            
            # Step 2: Tokenize
            inputs = self.model.processor(text=processed_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            results["input_shape"] = list(input_ids.shape)
            
            # Step 3: Get speaker embeddings
            speaker_embeddings = self.model._get_speaker_embeddings(voice).to(self.model.device)
            results["speaker_embeddings_shape"] = list(speaker_embeddings.shape)
            
            # Step 4: Generate speech with detailed monitoring
            with torch.inference_mode():
                # Monitor intermediate outputs
                logger.debug("Starting speech generation...")
                
                speech = self.model.model.generate_speech(
                    input_ids,
                    speaker_embeddings,
                    vocoder=self.model.vocoder
                )
                
                results["output_shape"] = list(speech.shape)
                results["output_dtype"] = str(speech.dtype)
                results["output_device"] = str(speech.device)
                
                # Audio statistics
                audio_np = speech.detach().cpu().numpy()
                results["audio_stats"] = {
                    "min": float(audio_np.min()),
                    "max": float(audio_np.max()),
                    "mean": float(audio_np.mean()),
                    "std": float(audio_np.std()),
                    "rms": float(np.sqrt(np.mean(audio_np.square()))),
                    "duration_seconds": len(audio_np) / self.model.SAMPLE_RATE,
                    "has_nan": bool(np.isnan(audio_np).any()),
                    "has_inf": bool(np.isinf(audio_np).any()),
                    "is_silent": bool(np.abs(audio_np).max() < 1e-6)
                }
                
                # Save audio sample for inspection
                temp_file = Path("temp") / f"technical_analysis_{voice}_{len(text)}chars.wav"
                temp_file.parent.mkdir(exist_ok=True)
                sf.write(str(temp_file), audio_np, self.model.SAMPLE_RATE)
                results["audio_file"] = str(temp_file)
                
                # Quick transcription test
                try:
                    validation_result = self.whisper_validator.validate_tts_output(
                        original_text=text,
                        audio_data=audio_np,
                        sample_rate=self.model.SAMPLE_RATE
                    )
                    results["transcription"] = validation_result.get("transcription", "")
                    results["accuracy"] = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                except Exception as e:
                    results["transcription_error"] = str(e)
                
            logger.info(f"Audio generation analysis complete - duration: {results['audio_stats']['duration_seconds']:.2f}s")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Audio generation analysis failed: {e}")
            traceback.print_exc()
        
        return results
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive technical analysis."""
        logger.info("🚀 Starting Comprehensive Technical Root Cause Analysis")
        logger.info("=" * 70)
        
        await self.initialize()
        
        # Test cases
        test_cases = [
            "Hello",
            "Hello world",
            "The quick brown fox jumps over the lazy dog."
        ]
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "torch_version": torch.__version__,
                "device": str(self.model.device),
                "model_type": type(self.model).__name__
            },
            "test_cases": {}
        }
        
        for i, text in enumerate(test_cases):
            logger.info(f"\n{'='*50}")
            logger.info(f"ANALYZING TEST CASE {i+1}: '{text}'")
            logger.info(f"{'='*50}")
            
            case_results = {}
            
            # 1. Text preprocessing analysis
            case_results["preprocessing"] = self.analyze_text_preprocessing(text)
            
            # 2. Model input analysis
            case_results["model_inputs"] = self.analyze_model_inputs(text)
            
            # 3. Speaker embeddings analysis
            case_results["speaker_embeddings"] = self.analyze_speaker_embeddings("alloy")
            
            # 4. Memory usage analysis
            case_results["memory_usage"] = self.analyze_memory_usage()
            
            # 5. Audio generation analysis
            case_results["audio_generation"] = self.analyze_audio_generation(text, "alloy")
            
            analysis_results["test_cases"][f"case_{i+1}_{len(text)}chars"] = case_results
            
            # Force garbage collection between tests
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Overall analysis
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL ANALYSIS")
        logger.info(f"{'='*70}")
        
        # Identify patterns
        patterns = self.identify_patterns(analysis_results)
        analysis_results["patterns"] = patterns
        
        # Save results
        output_file = Path("temp") / "technical_root_cause_analysis.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"📁 Technical analysis saved to: {output_file}")
        
        return analysis_results
    
    def identify_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in the analysis results."""
        patterns = {}
        
        # Check for consistent issues across test cases
        test_cases = results.get("test_cases", {})
        
        if test_cases:
            # Audio generation patterns
            silent_outputs = 0
            transcription_failures = 0
            low_accuracy_count = 0
            
            for case_name, case_data in test_cases.items():
                audio_gen = case_data.get("audio_generation", {})
                audio_stats = audio_gen.get("audio_stats", {})
                
                if audio_stats.get("is_silent", False):
                    silent_outputs += 1
                
                if "transcription_error" in audio_gen:
                    transcription_failures += 1
                
                accuracy = audio_gen.get("accuracy", 0)
                if accuracy < 10:  # Less than 10% accuracy
                    low_accuracy_count += 1
            
            patterns["audio_issues"] = {
                "silent_outputs": silent_outputs,
                "transcription_failures": transcription_failures,
                "low_accuracy_count": low_accuracy_count,
                "total_cases": len(test_cases)
            }
            
            # Preprocessing patterns
            phonemization_disabled = all(
                case_data.get("preprocessing", {}).get("preprocessing_steps", {}).get("6_phonemization") == "DISABLED"
                for case_data in test_cases.values()
            )
            patterns["phonemization_disabled"] = phonemization_disabled
        
        logger.info(f"Pattern analysis: {patterns}")
        return patterns


async def main():
    """Main execution function."""
    analyzer = TechnicalAnalyzer()
    results = await analyzer.run_comprehensive_analysis()
    
    patterns = results.get("patterns", {})
    audio_issues = patterns.get("audio_issues", {})
    
    logger.info(f"\n🎯 TECHNICAL ANALYSIS COMPLETE")
    logger.info(f"Silent outputs: {audio_issues.get('silent_outputs', 0)}/{audio_issues.get('total_cases', 0)}")
    logger.info(f"Low accuracy cases: {audio_issues.get('low_accuracy_count', 0)}/{audio_issues.get('total_cases', 0)}")
    logger.info(f"Phonemization disabled: {patterns.get('phonemization_disabled', False)}")
    
    if audio_issues.get('silent_outputs', 0) > 0:
        logger.error("❌ CRITICAL: Silent audio outputs detected")
    if audio_issues.get('low_accuracy_count', 0) == audio_issues.get('total_cases', 0):
        logger.error("❌ CRITICAL: All test cases show poor intelligibility")


if __name__ == "__main__":
    asyncio.run(main())
