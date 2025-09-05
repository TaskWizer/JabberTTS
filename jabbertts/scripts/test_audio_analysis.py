#!/usr/bin/env python3
"""
Test Audio Analysis Endpoint

This script tests the new audio analysis endpoint that provides:
- Waveform visualization data
- Spectrogram analysis
- Phoneme alignment markers
- Audio quality metrics

Usage:
    python test_audio_analysis.py
"""

import json
import logging
import requests
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_audio_analysis_endpoint(base_url: str = "http://localhost:8002"):
    """Test the audio analysis endpoint."""
    
    test_cases = [
        {
            "text": "Hello world",
            "voice": "alloy",
            "format": "wav"
        },
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "voice": "fable", 
            "format": "wav"
        },
        {
            "text": "Testing text-to-speech with phoneme analysis",
            "voice": "echo",
            "format": "wav"
        }
    ]
    
    results = {
        "test_suite": "audio_analysis",
        "timestamp": "2025-09-05",
        "server_url": base_url,
        "tests": [],
        "summary": {
            "total_tests": len(test_cases),
            "successful_tests": 0,
            "failed_tests": 0
        }
    }
    
    # Wait for server to be ready
    logger.info("Waiting for server to be ready...")
    for _ in range(30):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Server is ready")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    else:
        logger.error("Server not ready")
        return results
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Running test {i+1}: '{test_case['text'][:30]}...'")
        
        test_result = {
            "test_id": f"audio_analysis_test_{i+1}",
            "input": test_case,
            "success": False,
            "error": None,
            "analysis_data": {}
        }
        
        try:
            # Test the audio analysis endpoint
            data = {
                "text": test_case["text"],
                "voice": test_case["voice"],
                "format": test_case["format"],
                "speed": "1.0",
                "include_waveform": "true",
                "include_spectrogram": "true",
                "include_phonemes": "true"
            }
            
            response = requests.post(
                f"{base_url}/dashboard/debug/audio-analysis",
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data.get("success"):
                    test_result["success"] = True
                    
                    # Extract key analysis data
                    analysis = {}
                    
                    # Generation info
                    if "generation_info" in result_data:
                        gen_info = result_data["generation_info"]
                        analysis["generation"] = {
                            "rtf": gen_info.get("rtf", 0),
                            "inference_time": gen_info.get("inference_time", 0),
                            "audio_duration": gen_info.get("audio_duration", 0),
                            "sample_rate": gen_info.get("sample_rate", 0)
                        }
                    
                    # Phoneme analysis
                    if "phoneme_analysis" in result_data and not result_data["phoneme_analysis"].get("error"):
                        phoneme_info = result_data["phoneme_analysis"]
                        analysis["phonemes"] = {
                            "original_text": phoneme_info.get("original_text", ""),
                            "phonemized_text": phoneme_info.get("phonemized_text", ""),
                            "phoneme_count": phoneme_info.get("phoneme_count", 0),
                            "complexity_score": phoneme_info.get("complexity_score", 0)
                        }
                    
                    # Waveform data
                    if "waveform" in result_data and not result_data["waveform"].get("error"):
                        waveform_info = result_data["waveform"]
                        analysis["waveform"] = {
                            "duration": waveform_info.get("duration", 0),
                            "sample_rate": waveform_info.get("sample_rate", 0),
                            "data_points": len(waveform_info.get("amplitude", [])),
                            "downsample_factor": waveform_info.get("downsample_factor", 1)
                        }
                    
                    # Spectrogram data
                    if "spectrogram" in result_data and not result_data["spectrogram"].get("error"):
                        spec_info = result_data["spectrogram"]
                        analysis["spectrogram"] = {
                            "frequency_bins": len(spec_info.get("frequencies", [])),
                            "time_bins": len(spec_info.get("times", [])),
                            "freq_downsample": spec_info.get("freq_downsample", 1),
                            "time_downsample": spec_info.get("time_downsample", 1)
                        }
                    
                    # Quality analysis
                    if "quality_analysis" in result_data and not result_data["quality_analysis"].get("error"):
                        quality_info = result_data["quality_analysis"]
                        analysis["quality"] = {
                            "overall_score": quality_info.get("overall_score", 0),
                            "metrics": quality_info.get("metrics", {}),
                            "validation": quality_info.get("validation", {})
                        }
                    
                    test_result["analysis_data"] = analysis
                    results["summary"]["successful_tests"] += 1
                    
                    logger.info(f"✅ Test {i+1} passed")
                    logger.info(f"   RTF: {analysis.get('generation', {}).get('rtf', 0):.3f}")
                    logger.info(f"   Quality: {analysis.get('quality', {}).get('overall_score', 0):.1f}%")
                    logger.info(f"   Phonemes: {analysis.get('phonemes', {}).get('phoneme_count', 0)}")
                    
                else:
                    test_result["error"] = result_data.get("error", "Unknown error")
                    results["summary"]["failed_tests"] += 1
                    logger.error(f"❌ Test {i+1} failed: {test_result['error']}")
            else:
                test_result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                results["summary"]["failed_tests"] += 1
                logger.error(f"❌ Test {i+1} failed: {test_result['error']}")
                
        except Exception as e:
            test_result["error"] = str(e)
            results["summary"]["failed_tests"] += 1
            logger.error(f"❌ Test {i+1} exception: {e}")
        
        results["tests"].append(test_result)
    
    return results


def main():
    """Main test execution."""
    print("🧪 Starting Audio Analysis Endpoint Test")
    print("=" * 50)
    
    try:
        results = test_audio_analysis_endpoint()
        
        # Save results
        output_file = Path("audio_analysis_test_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        summary = results["summary"]
        print(f"\n📊 TEST RESULTS SUMMARY")
        print(f"{'='*30}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['successful_tests']/summary['total_tests']:.1%}")
        
        # Print detailed results
        print(f"\n🔍 DETAILED ANALYSIS RESULTS")
        print(f"{'='*40}")
        
        for test in results["tests"]:
            if test["success"]:
                analysis = test["analysis_data"]
                print(f"✅ {test['test_id']}")
                print(f"   Text: {test['input']['text']}")
                print(f"   Voice: {test['input']['voice']}")
                
                if "generation" in analysis:
                    gen = analysis["generation"]
                    print(f"   RTF: {gen['rtf']:.3f}")
                    print(f"   Duration: {gen['audio_duration']:.2f}s")
                    print(f"   Sample Rate: {gen['sample_rate']}Hz")
                
                if "phonemes" in analysis:
                    phonemes = analysis["phonemes"]
                    print(f"   Phonemes: {phonemes['phoneme_count']}")
                    print(f"   Complexity: {phonemes['complexity_score']:.3f}")
                
                if "quality" in analysis:
                    quality = analysis["quality"]
                    print(f"   Quality Score: {quality['overall_score']:.1f}%")
                
                if "waveform" in analysis:
                    waveform = analysis["waveform"]
                    print(f"   Waveform Points: {waveform['data_points']}")
                
                if "spectrogram" in analysis:
                    spec = analysis["spectrogram"]
                    print(f"   Spectrogram: {spec['frequency_bins']}×{spec['time_bins']}")
                
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
