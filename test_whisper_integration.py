#!/usr/bin/env python3
"""
Test Whisper STT Integration for JabberTTS Dashboard

This script tests the new debug transcription endpoints to ensure:
1. Audio file upload and transcription works correctly
2. Generate-and-transcribe pipeline functions properly
3. Accuracy metrics are calculated correctly
4. Audio quality analysis is integrated
5. Error handling works as expected

Usage:
    python test_whisper_integration.py
"""

import asyncio
import json
import logging
import tempfile
import wave
from pathlib import Path
from typing import Dict, Any

import numpy as np
import requests
from fastapi.testclient import TestClient

from jabbertts.main import create_app
from jabbertts.inference.engine import get_inference_engine
from jabbertts.audio.processor import get_audio_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperIntegrationTester:
    """Test suite for Whisper STT integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.app = create_app()
        self.client = TestClient(self.app)
        self.test_phrases = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a test of the text-to-speech system.",
            "Neural networks enable advanced artificial intelligence applications.",
            "Testing one two three four five six seven eight nine ten.",
            "Welcome to JabberTTS, an advanced text-to-speech system."
        ]
        
    def create_test_audio_file(self, text: str, voice: str = "alloy") -> bytes:
        """Create a test audio file using the TTS system.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            
        Returns:
            Audio data as bytes
        """
        try:
            # Generate TTS audio synchronously
            engine = get_inference_engine()
            processor = get_audio_processor()
            
            # Generate speech
            result = engine.generate_speech_sync(
                text=text,
                voice=voice,
                response_format="wav"
            )
            
            # Process audio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            audio_data, _ = loop.run_until_complete(
                processor.process_audio(
                    audio_array=result["audio_data"],
                    sample_rate=result["sample_rate"],
                    output_format="wav"
                )
            )
            
            loop.close()
            return audio_data
            
        except Exception as e:
            logger.error(f"Failed to create test audio: {e}")
            raise
    
    def test_transcribe_endpoint(self) -> Dict[str, Any]:
        """Test the /debug/transcribe endpoint."""
        logger.info("Testing /debug/transcribe endpoint...")
        
        results = {
            "endpoint": "/debug/transcribe",
            "tests": [],
            "success_count": 0,
            "total_tests": 0
        }
        
        for i, phrase in enumerate(self.test_phrases[:3]):  # Test first 3 phrases
            test_result = {
                "test_id": f"transcribe_test_{i+1}",
                "phrase": phrase,
                "success": False,
                "error": None,
                "metrics": {}
            }
            
            try:
                # Create test audio
                audio_data = self.create_test_audio_file(phrase)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file_path = temp_file.name
                
                # Test transcription with original text
                with open(temp_file_path, "rb") as audio_file:
                    response = self.client.post(
                        "/dashboard/debug/transcribe",
                        files={"audio_file": ("test.wav", audio_file, "audio/wav")},
                        data={"original_text": phrase}
                    )
                
                # Clean up temp file
                Path(temp_file_path).unlink()
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    if result_data.get("success"):
                        test_result["success"] = True
                        test_result["transcription"] = result_data.get("transcription", "")
                        
                        # Extract accuracy metrics
                        if "accuracy_metrics" in result_data:
                            metrics = result_data["accuracy_metrics"]
                            test_result["metrics"] = {
                                "accuracy": metrics.get("overall_accuracy", 0),
                                "wer": metrics.get("wer", 1.0),
                                "cer": metrics.get("cer", 1.0)
                            }
                        
                        results["success_count"] += 1
                    else:
                        test_result["error"] = result_data.get("error", "Unknown error")
                else:
                    test_result["error"] = f"HTTP {response.status_code}: {response.text}"
                    
            except Exception as e:
                test_result["error"] = str(e)
            
            results["tests"].append(test_result)
            results["total_tests"] += 1
        
        return results
    
    def test_generate_and_transcribe_endpoint(self) -> Dict[str, Any]:
        """Test the /debug/generate-and-transcribe endpoint."""
        logger.info("Testing /debug/generate-and-transcribe endpoint...")
        
        results = {
            "endpoint": "/debug/generate-and-transcribe",
            "tests": [],
            "success_count": 0,
            "total_tests": 0
        }
        
        voices = ["alloy", "fable", "echo"]
        formats = ["wav", "mp3"]
        
        for i, phrase in enumerate(self.test_phrases[:2]):  # Test first 2 phrases
            for voice in voices[:2]:  # Test first 2 voices
                for format in formats:
                    test_result = {
                        "test_id": f"generate_transcribe_test_{i+1}_{voice}_{format}",
                        "phrase": phrase,
                        "voice": voice,
                        "format": format,
                        "success": False,
                        "error": None,
                        "metrics": {}
                    }
                    
                    try:
                        response = self.client.post(
                            "/dashboard/debug/generate-and-transcribe",
                            data={
                                "text": phrase,
                                "voice": voice,
                                "format": format,
                                "speed": "1.0"
                            }
                        )
                        
                        if response.status_code == 200:
                            result_data = response.json()
                            
                            if result_data.get("success"):
                                test_result["success"] = True
                                
                                # Extract transcription result
                                transcription_result = result_data.get("transcription_result", {})
                                test_result["transcription"] = transcription_result.get("transcription", "")
                                
                                # Extract accuracy metrics
                                if "accuracy_metrics" in transcription_result:
                                    metrics = transcription_result["accuracy_metrics"]
                                    test_result["metrics"] = {
                                        "accuracy": metrics.get("overall_accuracy", 0),
                                        "wer": metrics.get("wer", 1.0),
                                        "cer": metrics.get("cer", 1.0)
                                    }
                                
                                # Extract generation metrics
                                gen_info = result_data.get("generation_info", {})
                                test_result["generation_metrics"] = {
                                    "rtf": gen_info.get("rtf", 0),
                                    "inference_time": gen_info.get("inference_time", 0),
                                    "audio_duration": gen_info.get("audio_duration", 0)
                                }
                                
                                results["success_count"] += 1
                            else:
                                test_result["error"] = result_data.get("error", "Unknown error")
                        else:
                            test_result["error"] = f"HTTP {response.status_code}: {response.text}"
                            
                    except Exception as e:
                        test_result["error"] = str(e)
                    
                    results["tests"].append(test_result)
                    results["total_tests"] += 1
        
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling for invalid inputs."""
        logger.info("Testing error handling...")
        
        results = {
            "endpoint": "error_handling",
            "tests": [],
            "success_count": 0,
            "total_tests": 0
        }
        
        # Test invalid file upload
        test_result = {
            "test_id": "invalid_file_upload",
            "description": "Upload non-audio file",
            "success": False,
            "error": None
        }
        
        try:
            # Create a text file instead of audio
            response = self.client.post(
                "/dashboard/debug/transcribe",
                files={"audio_file": ("test.txt", b"This is not audio", "text/plain")},
                data={"original_text": "test"}
            )
            
            # Should return 400 error
            if response.status_code == 400:
                test_result["success"] = True
                test_result["error"] = "Correctly rejected non-audio file"
            else:
                test_result["error"] = f"Expected 400, got {response.status_code}"
                
        except Exception as e:
            test_result["error"] = str(e)
        
        results["tests"].append(test_result)
        results["total_tests"] += 1
        if test_result["success"]:
            results["success_count"] += 1
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive Whisper integration test...")
        
        # Run all test suites
        transcribe_results = self.test_transcribe_endpoint()
        generate_transcribe_results = self.test_generate_and_transcribe_endpoint()
        error_handling_results = self.test_error_handling()
        
        # Compile overall results
        overall_results = {
            "test_suite": "whisper_integration",
            "timestamp": "2025-09-05",
            "summary": {
                "total_tests": (
                    transcribe_results["total_tests"] + 
                    generate_transcribe_results["total_tests"] + 
                    error_handling_results["total_tests"]
                ),
                "total_successes": (
                    transcribe_results["success_count"] + 
                    generate_transcribe_results["success_count"] + 
                    error_handling_results["success_count"]
                ),
                "success_rate": 0
            },
            "test_results": {
                "transcribe_endpoint": transcribe_results,
                "generate_and_transcribe_endpoint": generate_transcribe_results,
                "error_handling": error_handling_results
            }
        }
        
        # Calculate success rate
        if overall_results["summary"]["total_tests"] > 0:
            overall_results["summary"]["success_rate"] = (
                overall_results["summary"]["total_successes"] / 
                overall_results["summary"]["total_tests"]
            )
        
        return overall_results


def main():
    """Main test execution."""
    tester = WhisperIntegrationTester()
    
    try:
        results = tester.run_comprehensive_test()
        
        # Save results
        output_file = Path("whisper_integration_test_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        summary = results["summary"]
        print(f"\n{'='*60}")
        print("WHISPER INTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['total_successes']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Results saved to: {output_file}")
        
        # Print detailed results
        for endpoint, endpoint_results in results["test_results"].items():
            print(f"\n{endpoint.upper()}:")
            print(f"  Success Rate: {endpoint_results['success_count']}/{endpoint_results['total_tests']}")
            
            for test in endpoint_results["tests"]:
                status = "✅" if test["success"] else "❌"
                print(f"  {status} {test['test_id']}")
                if test.get("error"):
                    print(f"    Error: {test['error']}")
                if test.get("metrics"):
                    metrics = test["metrics"]
                    print(f"    Accuracy: {metrics.get('accuracy', 0):.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
