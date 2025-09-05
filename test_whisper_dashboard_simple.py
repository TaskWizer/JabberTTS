#!/usr/bin/env python3
"""
Simple test for Whisper STT Dashboard Integration

This script tests the new debug transcription functionality by:
1. Starting the JabberTTS server
2. Testing the dashboard endpoints with real HTTP requests
3. Validating transcription accuracy

Usage:
    python test_whisper_dashboard_simple.py
"""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleWhisperTester:
    """Simple tester for Whisper dashboard integration."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        """Initialize the tester.
        
        Args:
            base_url: Base URL of the JabberTTS server
        """
        self.base_url = base_url
        self.test_phrases = [
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing text-to-speech with Whisper validation."
        ]
    
    def wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for the server to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server is ready, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Server is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        logger.error("Server not ready within timeout")
        return False
    
    def test_dashboard_access(self) -> Dict[str, Any]:
        """Test that the dashboard is accessible."""
        logger.info("Testing dashboard access...")
        
        try:
            response = requests.get(f"{self.base_url}/dashboard/", timeout=10)
            
            if response.status_code == 200:
                # Check if the new debug transcription section is present
                content = response.text
                has_transcription_section = "Debug Transcription System" in content
                has_upload_form = "transcribeForm" in content
                has_generate_form = "generateTranscribeForm" in content
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "has_transcription_section": has_transcription_section,
                    "has_upload_form": has_upload_form,
                    "has_generate_form": has_generate_form,
                    "content_length": len(content)
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_generate_and_transcribe(self) -> Dict[str, Any]:
        """Test the generate-and-transcribe endpoint."""
        logger.info("Testing generate-and-transcribe endpoint...")
        
        results = {
            "endpoint": "generate-and-transcribe",
            "tests": [],
            "success_count": 0,
            "total_tests": 0
        }
        
        for i, phrase in enumerate(self.test_phrases):
            test_result = {
                "test_id": f"generate_test_{i+1}",
                "phrase": phrase,
                "success": False,
                "error": None,
                "metrics": {}
            }
            
            try:
                # Test the generate-and-transcribe endpoint
                data = {
                    "text": phrase,
                    "voice": "alloy",
                    "format": "wav",
                    "speed": "1.0"
                }
                
                response = requests.post(
                    f"{self.base_url}/dashboard/debug/generate-and-transcribe",
                    data=data,
                    timeout=30
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
                        logger.info(f"✅ Test {i+1} passed - Accuracy: {test_result['metrics'].get('accuracy', 0):.1f}%")
                    else:
                        test_result["error"] = result_data.get("error", "Unknown error")
                        logger.error(f"❌ Test {i+1} failed: {test_result['error']}")
                else:
                    test_result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.error(f"❌ Test {i+1} failed: {test_result['error']}")
                    
            except Exception as e:
                test_result["error"] = str(e)
                logger.error(f"❌ Test {i+1} exception: {e}")
            
            results["tests"].append(test_result)
            results["total_tests"] += 1
        
        return results
    
    def test_validation_endpoints(self) -> Dict[str, Any]:
        """Test existing validation endpoints still work."""
        logger.info("Testing existing validation endpoints...")
        
        endpoints_to_test = [
            "/dashboard/api/validation/health",
            "/dashboard/api/status",
            "/dashboard/api/performance"
        ]
        
        results = {
            "endpoint": "validation_endpoints",
            "tests": [],
            "success_count": 0,
            "total_tests": 0
        }
        
        for endpoint in endpoints_to_test:
            test_result = {
                "endpoint": endpoint,
                "success": False,
                "error": None,
                "response_data": None
            }
            
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    test_result["success"] = True
                    test_result["response_data"] = response.json()
                    results["success_count"] += 1
                    logger.info(f"✅ {endpoint} working")
                else:
                    test_result["error"] = f"HTTP {response.status_code}"
                    logger.error(f"❌ {endpoint} failed: {test_result['error']}")
                    
            except Exception as e:
                test_result["error"] = str(e)
                logger.error(f"❌ {endpoint} exception: {e}")
            
            results["tests"].append(test_result)
            results["total_tests"] += 1
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive Whisper dashboard test...")
        
        # Check if server is ready
        if not self.wait_for_server():
            return {
                "error": "Server not ready",
                "success": False
            }
        
        # Run all test suites
        dashboard_results = self.test_dashboard_access()
        generate_transcribe_results = self.test_generate_and_transcribe()
        validation_results = self.test_validation_endpoints()
        
        # Compile overall results
        overall_results = {
            "test_suite": "whisper_dashboard_integration",
            "timestamp": "2025-09-05",
            "server_url": self.base_url,
            "summary": {
                "dashboard_accessible": dashboard_results.get("success", False),
                "transcription_ui_present": dashboard_results.get("has_transcription_section", False),
                "generate_transcribe_success_rate": 0,
                "validation_endpoints_working": 0
            },
            "test_results": {
                "dashboard_access": dashboard_results,
                "generate_and_transcribe": generate_transcribe_results,
                "validation_endpoints": validation_results
            }
        }
        
        # Calculate success rates
        if generate_transcribe_results["total_tests"] > 0:
            overall_results["summary"]["generate_transcribe_success_rate"] = (
                generate_transcribe_results["success_count"] / 
                generate_transcribe_results["total_tests"]
            )
        
        if validation_results["total_tests"] > 0:
            overall_results["summary"]["validation_endpoints_working"] = (
                validation_results["success_count"] / 
                validation_results["total_tests"]
            )
        
        return overall_results


def main():
    """Main test execution."""
    print("🧪 Starting Whisper Dashboard Integration Test")
    print("=" * 60)
    
    tester = SimpleWhisperTester()
    
    try:
        results = tester.run_comprehensive_test()
        
        # Save results
        output_file = Path("whisper_dashboard_test_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            return
        
        summary = results["summary"]
        print(f"\n📊 TEST RESULTS SUMMARY")
        print(f"{'='*40}")
        print(f"Dashboard Accessible: {'✅' if summary['dashboard_accessible'] else '❌'}")
        print(f"Transcription UI Present: {'✅' if summary['transcription_ui_present'] else '❌'}")
        print(f"Generate & Transcribe Success Rate: {summary['generate_transcribe_success_rate']:.1%}")
        print(f"Validation Endpoints Working: {summary['validation_endpoints_working']:.1%}")
        
        # Print detailed results for generate-and-transcribe tests
        gen_results = results["test_results"]["generate_and_transcribe"]
        print(f"\n🎯 TRANSCRIPTION ACCURACY RESULTS")
        print(f"{'='*40}")
        
        for test in gen_results["tests"]:
            if test["success"]:
                metrics = test.get("metrics", {})
                gen_metrics = test.get("generation_metrics", {})
                print(f"✅ {test['test_id']}")
                print(f"   Original: {test['phrase']}")
                print(f"   Transcribed: {test.get('transcription', 'N/A')}")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.1f}%")
                print(f"   WER: {metrics.get('wer', 1.0):.3f}")
                print(f"   RTF: {gen_metrics.get('rtf', 0):.3f}")
            else:
                print(f"❌ {test['test_id']}: {test.get('error', 'Unknown error')}")
        
        print(f"\n📁 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
