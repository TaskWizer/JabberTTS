#!/usr/bin/env python3
"""
Model Checkpoint Investigation

This script investigates the SpeechT5 model checkpoint and vocoder
to identify if we're using the wrong model or incompatible components.

Usage:
    python jabbertts/scripts/investigate_model_checkpoint.py
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

from jabbertts.validation.whisper_validator import get_whisper_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def investigate_model_checkpoint():
    """Investigate SpeechT5 model checkpoint and components."""
    logger.info("🔍 Investigating SpeechT5 Model Checkpoint and Components")
    logger.info("=" * 70)
    
    try:
        whisper_validator = get_whisper_validator("base")
        
        # Test different model configurations
        test_configs = [
            {
                "name": "Standard SpeechT5 (Current)",
                "model_id": "microsoft/speecht5_tts",
                "vocoder_id": "microsoft/speecht5_hifigan"
            },
            {
                "name": "Alternative Vocoder Test",
                "model_id": "microsoft/speecht5_tts", 
                "vocoder_id": "microsoft/speecht5_hifigan",
                "use_alternative_loading": True
            }
        ]
        
        results = {
            "checkpoint_investigation": {
                "timestamp": datetime.now().isoformat(),
                "configurations_tested": [],
                "best_result": None,
                "recommendations": []
            }
        }
        
        for config in test_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"TESTING: {config['name']}")
            logger.info(f"{'='*60}")
            logger.info(f"Model: {config['model_id']}")
            logger.info(f"Vocoder: {config['vocoder_id']}")
            
            config_result = {
                "config": config,
                "model_info": {},
                "test_results": [],
                "errors": []
            }
            
            try:
                # Load components
                logger.info("Loading model components...")
                
                processor = SpeechT5Processor.from_pretrained(config["model_id"])
                model = SpeechT5ForTextToSpeech.from_pretrained(config["model_id"])
                vocoder = SpeechT5HifiGan.from_pretrained(config["vocoder_id"])
                
                # Set to eval mode
                model.eval()
                vocoder.eval()
                
                # Get model info
                config_result["model_info"] = {
                    "model_config": str(model.config),
                    "vocab_size": processor.tokenizer.vocab_size,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "vocoder_parameters": sum(p.numel() for p in vocoder.parameters())
                }
                
                logger.info(f"Model parameters: {config_result['model_info']['model_parameters']:,}")
                logger.info(f"Vocoder parameters: {config_result['model_info']['vocoder_parameters']:,}")
                logger.info(f"Vocab size: {config_result['model_info']['vocab_size']}")
                
                # Test with different texts
                test_texts = [
                    "Hello",
                    "Hello world",
                    "Testing speech synthesis"
                ]
                
                # Create simple speaker embeddings
                speaker_embeddings = torch.randn(1, 512) * 0.05  # Small random embeddings
                
                for test_text in test_texts:
                    logger.info(f"\nTesting: '{test_text}'")
                    
                    try:
                        # Tokenize
                        inputs = processor(text=test_text, return_tensors="pt")
                        input_ids = inputs["input_ids"]
                        
                        logger.info(f"  Input tokens: {input_ids}")
                        logger.info(f"  Token shape: {input_ids.shape}")
                        
                        # Generate speech
                        with torch.no_grad():
                            speech = model.generate_speech(
                                input_ids,
                                speaker_embeddings,
                                vocoder=vocoder
                            )
                        
                        logger.info(f"  Generated audio shape: {speech.shape}")
                        logger.info(f"  Audio range: [{speech.min():.4f}, {speech.max():.4f}]")
                        logger.info(f"  Audio std: {speech.std():.4f}")
                        
                        # Check for silence
                        if speech.abs().max() < 0.001:
                            logger.warning("  ⚠️ Generated audio is essentially silent!")
                            audio_status = "SILENT"
                        elif speech.std() < 0.01:
                            logger.warning("  ⚠️ Generated audio has very low variation")
                            audio_status = "LOW_VARIATION"
                        else:
                            logger.info("  ✅ Generated audio has reasonable properties")
                            audio_status = "NORMAL"
                        
                        # Save audio
                        audio_np = speech.detach().cpu().numpy()
                        audio_file = Path("temp") / f"checkpoint_test_{config['name'].replace(' ', '_').lower()}_{test_text.replace(' ', '_')}.wav"
                        sf.write(str(audio_file), audio_np, 16000)
                        logger.info(f"  Audio saved: {audio_file}")
                        
                        # Test transcription
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            sf.write(temp_file.name, audio_np, 16000)
                            with open(temp_file.name, "rb") as f:
                                audio_bytes = f.read()
                        
                        validation_result = whisper_validator.validate_tts_output(
                            original_text=test_text,
                            audio_data=audio_bytes,
                            sample_rate=16000
                        )
                        
                        accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
                        transcription = validation_result.get("transcription", "")
                        
                        logger.info(f"  Transcription: '{transcription}'")
                        logger.info(f"  Accuracy: {accuracy:.1f}%")
                        
                        test_result = {
                            "text": test_text,
                            "transcription": transcription,
                            "accuracy": accuracy,
                            "audio_status": audio_status,
                            "audio_file": str(audio_file)
                        }
                        
                        config_result["test_results"].append(test_result)
                        
                        # Check for breakthrough
                        if accuracy > 50:
                            logger.info(f"  🎉 BREAKTHROUGH: Intelligible audio achieved!")
                            break
                            
                    except Exception as e:
                        logger.error(f"  Test failed: {e}")
                        config_result["errors"].append(f"Test '{test_text}' failed: {e}")
                
                # Calculate average accuracy for this config
                if config_result["test_results"]:
                    avg_accuracy = sum(r["accuracy"] for r in config_result["test_results"]) / len(config_result["test_results"])
                    max_accuracy = max(r["accuracy"] for r in config_result["test_results"])
                    config_result["avg_accuracy"] = avg_accuracy
                    config_result["max_accuracy"] = max_accuracy
                    
                    logger.info(f"\nConfiguration Summary:")
                    logger.info(f"  Average accuracy: {avg_accuracy:.1f}%")
                    logger.info(f"  Maximum accuracy: {max_accuracy:.1f}%")
                    logger.info(f"  Tests completed: {len(config_result['test_results'])}")
                    logger.info(f"  Errors: {len(config_result['errors'])}")
                
            except Exception as e:
                logger.error(f"Configuration failed: {e}")
                config_result["errors"].append(f"Configuration failed: {e}")
                import traceback
                traceback.print_exc()
            
            results["checkpoint_investigation"]["configurations_tested"].append(config_result)
        
        # Analyze results
        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS AND RECOMMENDATIONS")
        logger.info(f"{'='*70}")
        
        successful_configs = [c for c in results["checkpoint_investigation"]["configurations_tested"] 
                            if c.get("avg_accuracy", 0) > 0]
        
        if successful_configs:
            best_config = max(successful_configs, key=lambda x: x.get("max_accuracy", 0))
            results["checkpoint_investigation"]["best_result"] = best_config
            
            logger.info(f"Best configuration: {best_config['config']['name']}")
            logger.info(f"  Maximum accuracy: {best_config.get('max_accuracy', 0):.1f}%")
            logger.info(f"  Average accuracy: {best_config.get('avg_accuracy', 0):.1f}%")
            
            if best_config.get("max_accuracy", 0) > 50:
                logger.info("🎉 FOUND WORKING CONFIGURATION!")
                results["checkpoint_investigation"]["recommendations"].append(
                    f"Use configuration: {best_config['config']['name']}"
                )
            elif best_config.get("max_accuracy", 0) > 10:
                logger.info("✅ Found partially working configuration")
                results["checkpoint_investigation"]["recommendations"].append(
                    f"Investigate configuration: {best_config['config']['name']}"
                )
            else:
                logger.info("❌ No configuration achieved good intelligibility")
        else:
            logger.error("❌ All configurations failed")
        
        # General recommendations
        recommendations = [
            "Test with different SpeechT5 model checkpoints",
            "Verify speaker embeddings are compatible with model",
            "Check if model requires specific preprocessing",
            "Test with reference SpeechT5 implementations",
            "Consider using different TTS model (e.g., Tacotron2, FastSpeech2)"
        ]
        
        results["checkpoint_investigation"]["recommendations"].extend(recommendations)
        
        logger.info("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(results["checkpoint_investigation"]["recommendations"], 1):
            logger.info(f"  {i}. {rec}")
        
        # Save results
        output_file = Path("temp") / "model_checkpoint_investigation.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n📁 Investigation results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_reference_implementation():
    """Test a minimal reference SpeechT5 implementation."""
    logger.info("\n🧪 Testing Minimal Reference Implementation")
    logger.info("=" * 50)
    
    try:
        # This is the most basic SpeechT5 usage from the official docs
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset
        
        logger.info("Loading reference implementation...")
        
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        
        # Try to load speaker embeddings from the official dataset
        try:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            logger.info("✅ Loaded official speaker embeddings")
        except Exception as e:
            logger.warning(f"Could not load official embeddings: {e}")
            # Use random embeddings as fallback
            speaker_embeddings = torch.randn(1, 512) * 0.05
            logger.info("Using random speaker embeddings")
        
        # Test with simple text
        text = "Hello world"
        logger.info(f"Testing reference implementation with: '{text}'")
        
        inputs = processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
        logger.info(f"Reference audio shape: {speech.shape}")
        logger.info(f"Reference audio range: [{speech.min():.4f}, {speech.max():.4f}]")
        logger.info(f"Reference audio std: {speech.std():.4f}")
        
        # Save reference audio
        audio_np = speech.detach().cpu().numpy()
        ref_audio_file = Path("temp") / "reference_implementation.wav"
        sf.write(str(ref_audio_file), audio_np, 16000)
        logger.info(f"Reference audio saved: {ref_audio_file}")
        
        # Test transcription
        whisper_validator = get_whisper_validator("base")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_np, 16000)
            with open(temp_file.name, "rb") as f:
                audio_bytes = f.read()
        
        validation_result = whisper_validator.validate_tts_output(
            original_text=text,
            audio_data=audio_bytes,
            sample_rate=16000
        )
        
        accuracy = validation_result.get("accuracy_metrics", {}).get("overall_accuracy", 0)
        transcription = validation_result.get("transcription", "")
        
        logger.info(f"Reference transcription: '{transcription}'")
        logger.info(f"Reference accuracy: {accuracy:.1f}%")
        
        if accuracy > 50:
            logger.info("🎉 REFERENCE IMPLEMENTATION WORKS!")
            logger.info("The issue is in our implementation, not the model")
        elif accuracy > 10:
            logger.info("✅ Reference implementation partially works")
            logger.info("Our implementation may have additional issues")
        else:
            logger.info("❌ Reference implementation also fails")
            logger.info("The issue may be fundamental to SpeechT5 or environment")
        
        return {
            "reference_test": {
                "text": text,
                "transcription": transcription,
                "accuracy": accuracy,
                "audio_file": str(ref_audio_file)
            }
        }
        
    except Exception as e:
        logger.error(f"Reference implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main execution function."""
    logger.info("🔍 COMPREHENSIVE MODEL CHECKPOINT INVESTIGATION")
    logger.info("=" * 70)
    logger.info("Investigating if we're using wrong model weights or incompatible components")
    logger.info("=" * 70)
    
    # Test our configurations
    checkpoint_results = await investigate_model_checkpoint()
    
    # Test reference implementation
    reference_results = await test_reference_implementation()
    
    # Combine results
    if checkpoint_results and reference_results:
        checkpoint_results.update(reference_results)
    
    logger.info("\n🎯 INVESTIGATION COMPLETE")
    logger.info("Check temp/ directory for audio samples and detailed results")
    
    return checkpoint_results


if __name__ == "__main__":
    asyncio.run(main())
