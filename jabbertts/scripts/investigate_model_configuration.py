#!/usr/bin/env python3
"""
Model Configuration Investigation

This script investigates the SpeechT5 model configuration to identify
deeper issues beyond phonemization that cause unintelligible audio.

Usage:
    python jabbertts/scripts/investigate_model_configuration.py
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import soundfile as sf

from jabbertts.inference.engine import get_inference_engine
from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def investigate_model_configuration():
    """Investigate SpeechT5 model configuration issues."""
    logger.info("🔍 Investigating SpeechT5 Model Configuration")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        inference_engine = get_inference_engine()
        whisper_validator = get_whisper_validator("base")
        
        # Load the model
        model = await inference_engine._ensure_model_loaded()
        
        logger.info("📋 MODEL INFORMATION")
        logger.info("=" * 30)
        logger.info(f"Model class: {type(model).__name__}")
        logger.info(f"Model name: {getattr(model, 'name', 'Unknown')}")
        logger.info(f"Sample rate: {model.get_sample_rate()}")
        logger.info(f"Device: {getattr(model, 'device', 'Unknown')}")
        logger.info("")
        
        # Test 1: Examine model components
        logger.info("🔧 MODEL COMPONENTS ANALYSIS")
        logger.info("=" * 30)
        
        if hasattr(model, 'model'):
            logger.info(f"TTS Model: {type(model.model).__name__}")
            logger.info(f"Model config: {model.model.config}")
        
        if hasattr(model, 'processor'):
            logger.info(f"Processor: {type(model.processor).__name__}")
            logger.info(f"Tokenizer vocab size: {model.processor.tokenizer.vocab_size}")
        
        if hasattr(model, 'vocoder'):
            logger.info(f"Vocoder: {type(model.vocoder).__name__}")
        
        logger.info("")
        
        # Test 2: Examine speaker embeddings
        logger.info("🎭 SPEAKER EMBEDDINGS ANALYSIS")
        logger.info("=" * 30)
        
        test_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        for voice in test_voices:
            try:
                if hasattr(model, '_get_speaker_embeddings'):
                    embeddings = model._get_speaker_embeddings(voice)
                    logger.info(f"Voice '{voice}': shape={embeddings.shape}, mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
                else:
                    logger.info(f"Voice '{voice}': No speaker embeddings method found")
            except Exception as e:
                logger.error(f"Voice '{voice}': Error getting embeddings - {e}")
        
        logger.info("")
        
        # Test 3: Raw model inference test
        logger.info("🧪 RAW MODEL INFERENCE TEST")
        logger.info("=" * 30)
        
        test_text = "Hello"
        logger.info(f"Testing with: '{test_text}'")
        
        try:
            # Test tokenization
            inputs = model.processor(text=test_text, return_tensors="pt")
            logger.info(f"Input tokens: {inputs['input_ids']}")
            logger.info(f"Token shape: {inputs['input_ids'].shape}")
            
            # Test speaker embeddings
            speaker_embeddings = model._get_speaker_embeddings("alloy")
            logger.info(f"Speaker embeddings shape: {speaker_embeddings.shape}")
            logger.info(f"Speaker embeddings range: [{speaker_embeddings.min():.4f}, {speaker_embeddings.max():.4f}]")
            
            # Test raw model generation
            with torch.inference_mode():
                input_ids = inputs["input_ids"].to(model.device)
                speaker_embeddings = speaker_embeddings.to(model.device)
                
                logger.info("Generating speech with raw model...")
                speech = model.model.generate_speech(
                    input_ids,
                    speaker_embeddings,
                    vocoder=model.vocoder
                )
                
                logger.info(f"Generated audio shape: {speech.shape}")
                logger.info(f"Audio range: [{speech.min():.4f}, {speech.max():.4f}]")
                logger.info(f"Audio mean: {speech.mean():.4f}")
                logger.info(f"Audio std: {speech.std():.4f}")
                
                # Check for silence or noise
                if speech.abs().max() < 0.001:
                    logger.critical("🚨 CRITICAL: Generated audio is essentially silent!")
                elif speech.std() < 0.01:
                    logger.warning("⚠️ WARNING: Generated audio has very low variation")
                else:
                    logger.info("✅ Generated audio has reasonable amplitude and variation")
                
                # Save raw audio for inspection
                audio_np = speech.detach().cpu().numpy()
                raw_audio_file = Path("temp") / "raw_model_output.wav"
                sf.write(str(raw_audio_file), audio_np, model.get_sample_rate())
                logger.info(f"Raw audio saved to: {raw_audio_file}")
                
                # Test transcription of raw audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file.name, audio_np, model.get_sample_rate())
                    with open(temp_file.name, "rb") as f:
                        audio_bytes = f.read()
                
                validation_result = whisper_validator.validate_tts_output(
                    original_text=test_text,
                    audio_data=audio_bytes,
                    sample_rate=model.get_sample_rate()
                )
                
                raw_accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                raw_transcription = validation_result.get("transcription", "")
                
                logger.info(f"Raw model transcription: '{raw_transcription}'")
                logger.info(f"Raw model accuracy: {raw_accuracy:.1f}%")
                
        except Exception as e:
            logger.error(f"Raw model test failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("")
        
        # Test 4: Compare with different texts
        logger.info("📝 TEXT COMPLEXITY TEST")
        logger.info("=" * 30)
        
        test_texts = [
            "A",
            "Hello",
            "Test",
            "One two three",
            "The cat sat on the mat"
        ]
        
        for text in test_texts:
            try:
                logger.info(f"Testing: '{text}'")
                
                # Generate with full pipeline
                result = await inference_engine.generate_speech(
                    text=text,
                    voice="alloy",
                    response_format="wav"
                )
                
                # Quick transcription
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file.name, result["audio_data"], result["sample_rate"])
                    with open(temp_file.name, "rb") as f:
                        audio_bytes = f.read()
                
                validation_result = whisper_validator.validate_tts_output(
                    original_text=text,
                    audio_data=audio_bytes,
                    sample_rate=result["sample_rate"]
                )
                
                accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                transcription = validation_result.get("transcription", "")
                
                logger.info(f"  Transcription: '{transcription}'")
                logger.info(f"  Accuracy: {accuracy:.1f}%")
                logger.info(f"  Audio duration: {len(result['audio_data'])/result['sample_rate']:.2f}s")
                logger.info(f"  Audio range: [{result['audio_data'].min():.4f}, {result['audio_data'].max():.4f}]")
                
                if accuracy > 50:
                    logger.info(f"  🎉 SUCCESS: Found intelligible output!")
                    break
                    
            except Exception as e:
                logger.error(f"  Error testing '{text}': {e}")
        
        logger.info("")
        
        # Test 5: Model parameter investigation
        logger.info("⚙️ MODEL PARAMETERS INVESTIGATION")
        logger.info("=" * 30)
        
        try:
            # Check model state
            if hasattr(model.model, 'training'):
                logger.info(f"Model training mode: {model.model.training}")
            
            # Check for NaN or inf in model parameters
            nan_params = 0
            inf_params = 0
            total_params = 0
            
            for name, param in model.model.named_parameters():
                total_params += 1
                if torch.isnan(param).any():
                    nan_params += 1
                    logger.warning(f"NaN found in parameter: {name}")
                if torch.isinf(param).any():
                    inf_params += 1
                    logger.warning(f"Inf found in parameter: {name}")
            
            logger.info(f"Total parameters: {total_params}")
            logger.info(f"Parameters with NaN: {nan_params}")
            logger.info(f"Parameters with Inf: {inf_params}")
            
            if nan_params > 0 or inf_params > 0:
                logger.critical("🚨 CRITICAL: Model has corrupted parameters!")
            else:
                logger.info("✅ Model parameters appear healthy")
                
        except Exception as e:
            logger.error(f"Parameter investigation failed: {e}")
        
        logger.info("")
        logger.info("🎯 INVESTIGATION COMPLETE")
        logger.info("=" * 30)
        
        # Summary and recommendations
        recommendations = []
        
        if raw_accuracy < 10:
            recommendations.append("Raw model output is unintelligible - check model weights")
        
        if nan_params > 0 or inf_params > 0:
            recommendations.append("Model has corrupted parameters - reload model")
        
        recommendations.extend([
            "Test with different SpeechT5 model checkpoint",
            "Verify vocoder configuration",
            "Check speaker embedding quality",
            "Test with minimal inference parameters"
        ])
        
        logger.info("📋 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        return {
            "model_investigation": {
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "class": type(model).__name__,
                    "sample_rate": model.get_sample_rate(),
                    "device": str(getattr(model, 'device', 'Unknown'))
                },
                "raw_model_accuracy": raw_accuracy if 'raw_accuracy' in locals() else 0,
                "parameter_health": {
                    "total_params": total_params if 'total_params' in locals() else 0,
                    "nan_params": nan_params if 'nan_params' in locals() else 0,
                    "inf_params": inf_params if 'inf_params' in locals() else 0
                },
                "recommendations": recommendations
            }
        }
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main execution function."""
    results = await investigate_model_configuration()
    
    if results:
        output_file = Path("temp") / "model_configuration_investigation.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"📁 Investigation results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
