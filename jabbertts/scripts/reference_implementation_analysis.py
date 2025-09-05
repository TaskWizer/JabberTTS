#!/usr/bin/env python3
"""
Reference Implementation Analysis

This script analyzes reference TTS implementations to understand proper
SpeechT5 configuration and identify differences with our implementation.

Usage:
    python jabbertts/scripts/reference_implementation_analysis.py
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReferenceImplementationAnalyzer:
    """Analyzes reference TTS implementations."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.repos_base = Path("/home/mkinney/Repos")
        self.analysis_results = {}
    
    def analyze_repository_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository structure and key files."""
        try:
            if not repo_path.exists():
                return {"error": f"Repository not found: {repo_path}"}
            
            analysis = {
                "repo_name": repo_path.name,
                "repo_path": str(repo_path),
                "exists": True,
                "key_files": {},
                "python_files": [],
                "config_files": [],
                "model_files": [],
                "requirements": {}
            }
            
            # Find key files
            key_patterns = {
                "main": ["main.py", "app.py", "server.py", "run.py"],
                "tts": ["tts.py", "speech.py", "synthesis.py", "generate.py"],
                "model": ["model.py", "models.py", "inference.py"],
                "config": ["config.py", "settings.py", "configuration.py"],
                "requirements": ["requirements.txt", "pyproject.toml", "setup.py", "environment.yml"]
            }
            
            for category, patterns in key_patterns.items():
                found_files = []
                for pattern in patterns:
                    matches = list(repo_path.rglob(pattern))
                    found_files.extend([str(f.relative_to(repo_path)) for f in matches])
                analysis["key_files"][category] = found_files
            
            # Find Python files
            python_files = list(repo_path.rglob("*.py"))
            analysis["python_files"] = [str(f.relative_to(repo_path)) for f in python_files[:20]]  # Limit to first 20
            
            # Find config files
            config_extensions = [".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"]
            for ext in config_extensions:
                config_files = list(repo_path.rglob(f"*{ext}"))
                analysis["config_files"].extend([str(f.relative_to(repo_path)) for f in config_files[:10]])
            
            # Check for model-related files
            model_extensions = [".pt", ".pth", ".onnx", ".bin", ".safetensors"]
            for ext in model_extensions:
                model_files = list(repo_path.rglob(f"*{ext}"))
                analysis["model_files"].extend([str(f.relative_to(repo_path)) for f in model_files[:5]])
            
            # Read requirements if available
            for req_file in ["requirements.txt", "pyproject.toml"]:
                req_path = repo_path / req_file
                if req_path.exists():
                    try:
                        with open(req_path, 'r') as f:
                            analysis["requirements"][req_file] = f.read()[:1000]  # First 1000 chars
                    except Exception as e:
                        analysis["requirements"][req_file] = f"Error reading: {e}"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {repo_path}: {e}")
            return {"error": str(e)}
    
    def search_for_speecht5_usage(self, repo_path: Path) -> Dict[str, Any]:
        """Search for SpeechT5 usage patterns in the repository."""
        try:
            speecht5_analysis = {
                "repo_name": repo_path.name,
                "speecht5_mentions": [],
                "model_loading": [],
                "inference_patterns": [],
                "configuration_patterns": []
            }
            
            if not repo_path.exists():
                return speecht5_analysis
            
            # Search for SpeechT5 mentions in Python files
            python_files = list(repo_path.rglob("*.py"))
            
            for py_file in python_files[:50]:  # Limit search
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Look for SpeechT5 mentions
                        if 'speecht5' in content.lower() or 'speech_t5' in content.lower():
                            speecht5_analysis["speecht5_mentions"].append({
                                "file": str(py_file.relative_to(repo_path)),
                                "lines": [i+1 for i, line in enumerate(content.split('\n')) 
                                         if 'speecht5' in line.lower() or 'speech_t5' in line.lower()][:10]
                            })
                        
                        # Look for model loading patterns
                        model_patterns = [
                            'from_pretrained',
                            'load_model',
                            'AutoModel',
                            'SpeechT5ForTextToSpeech',
                            'SpeechT5Processor'
                        ]
                        
                        for pattern in model_patterns:
                            if pattern in content:
                                lines = [i+1 for i, line in enumerate(content.split('\n')) 
                                        if pattern in line][:5]
                                if lines:
                                    speecht5_analysis["model_loading"].append({
                                        "file": str(py_file.relative_to(repo_path)),
                                        "pattern": pattern,
                                        "lines": lines
                                    })
                        
                        # Look for inference patterns
                        inference_patterns = [
                            'generate_speech',
                            'synthesize',
                            'forward',
                            'inference',
                            'text_to_speech'
                        ]
                        
                        for pattern in inference_patterns:
                            if pattern in content:
                                lines = [i+1 for i, line in enumerate(content.split('\n')) 
                                        if pattern in line][:3]
                                if lines:
                                    speecht5_analysis["inference_patterns"].append({
                                        "file": str(py_file.relative_to(repo_path)),
                                        "pattern": pattern,
                                        "lines": lines
                                    })
                
                except Exception as e:
                    continue  # Skip files that can't be read
            
            return speecht5_analysis
            
        except Exception as e:
            logger.error(f"Failed to search SpeechT5 usage in {repo_path}: {e}")
            return {"error": str(e)}
    
    def analyze_kokoro_fastapi(self) -> Dict[str, Any]:
        """Analyze Kokoro-FastAPI implementation."""
        logger.info("Analyzing Kokoro-FastAPI implementation...")
        
        repo_path = self.repos_base / "Kokoro-FastAPI"
        analysis = {
            "repository": "Kokoro-FastAPI",
            "structure": self.analyze_repository_structure(repo_path),
            "speecht5_usage": self.search_for_speecht5_usage(repo_path),
            "key_insights": []
        }
        
        # Look for specific configuration patterns
        if repo_path.exists():
            # Check main application files
            main_files = ["main.py", "app.py", "server.py"]
            for main_file in main_files:
                main_path = repo_path / main_file
                if main_path.exists():
                    try:
                        with open(main_path, 'r') as f:
                            content = f.read()
                            analysis["key_insights"].append({
                                "file": main_file,
                                "content_preview": content[:500],
                                "has_tts": "tts" in content.lower(),
                                "has_speech": "speech" in content.lower()
                            })
                    except Exception as e:
                        analysis["key_insights"].append({
                            "file": main_file,
                            "error": str(e)
                        })
        
        return analysis
    
    def analyze_styletts2(self) -> Dict[str, Any]:
        """Analyze StyleTTS2 implementation."""
        logger.info("Analyzing StyleTTS2 implementation...")
        
        repo_path = self.repos_base / "StyleTTS2"
        analysis = {
            "repository": "StyleTTS2",
            "structure": self.analyze_repository_structure(repo_path),
            "speecht5_usage": self.search_for_speecht5_usage(repo_path),
            "key_insights": []
        }
        
        # StyleTTS2 specific analysis
        if repo_path.exists():
            # Look for inference scripts
            inference_files = list(repo_path.rglob("*inference*"))
            demo_files = list(repo_path.rglob("*demo*"))
            
            analysis["key_insights"].append({
                "inference_files": [str(f.relative_to(repo_path)) for f in inference_files],
                "demo_files": [str(f.relative_to(repo_path)) for f in demo_files]
            })
        
        return analysis
    
    def analyze_litetts(self) -> Dict[str, Any]:
        """Analyze LiteTTS implementation."""
        logger.info("Analyzing LiteTTS implementation...")
        
        repo_path = self.repos_base / "LiteTTS"
        analysis = {
            "repository": "LiteTTS",
            "structure": self.analyze_repository_structure(repo_path),
            "speecht5_usage": self.search_for_speecht5_usage(repo_path),
            "key_insights": []
        }
        
        # LiteTTS specific analysis
        if repo_path.exists():
            # Look for backend implementations
            backends_path = repo_path / "LiteTTS" / "backends"
            if backends_path.exists():
                backend_files = list(backends_path.rglob("*.py"))
                analysis["key_insights"].append({
                    "backend_files": [str(f.relative_to(repo_path)) for f in backend_files],
                    "has_tts_cpp": any("TTS.cpp" in str(f) for f in backends_path.rglob("*"))
                })
        
        return analysis
    
    def analyze_espeak_integration(self) -> Dict[str, Any]:
        """Analyze eSpeak integration patterns."""
        logger.info("Analyzing eSpeak integration patterns...")
        
        analysis = {
            "repositories": ["espeak-ng", "espeak"],
            "integration_patterns": [],
            "phoneme_processing": []
        }
        
        for repo_name in ["espeak-ng", "espeak"]:
            repo_path = self.repos_base / repo_name
            if repo_path.exists():
                # Look for Python bindings or integration files
                python_files = list(repo_path.rglob("*.py"))
                analysis["integration_patterns"].append({
                    "repo": repo_name,
                    "python_files": [str(f.relative_to(repo_path)) for f in python_files[:10]],
                    "has_python_bindings": len(python_files) > 0
                })
        
        return analysis
    
    def compare_with_jabbertts(self) -> Dict[str, Any]:
        """Compare findings with JabberTTS implementation."""
        logger.info("Comparing with JabberTTS implementation...")
        
        jabbertts_path = Path("/home/mkinney/Repos/JabberTTS")
        
        comparison = {
            "jabbertts_analysis": {
                "structure": self.analyze_repository_structure(jabbertts_path),
                "speecht5_usage": self.search_for_speecht5_usage(jabbertts_path)
            },
            "differences": [],
            "recommendations": []
        }
        
        # Analyze key differences
        jabbertts_speecht5 = comparison["jabbertts_analysis"]["speecht5_usage"]
        
        # Check if we have proper SpeechT5 usage
        if not jabbertts_speecht5["speecht5_mentions"]:
            comparison["differences"].append("No explicit SpeechT5 mentions found in JabberTTS")
        
        if not jabbertts_speecht5["model_loading"]:
            comparison["differences"].append("No standard model loading patterns found")
        
        # Generate recommendations
        comparison["recommendations"] = [
            "Compare model loading patterns with reference implementations",
            "Verify SpeechT5 processor configuration",
            "Check speaker embedding handling",
            "Validate phoneme preprocessing pipeline",
            "Review inference parameter settings"
        ]
        
        return comparison
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of all reference implementations."""
        logger.info("🔍 Starting Comprehensive Reference Implementation Analysis")
        logger.info("=" * 60)
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "reference_implementation_comparison",
            "repositories_analyzed": [],
            "key_findings": [],
            "critical_differences": [],
            "recommendations": []
        }
        
        # Analyze each reference implementation
        implementations = [
            ("Kokoro-FastAPI", self.analyze_kokoro_fastapi),
            ("StyleTTS2", self.analyze_styletts2),
            ("LiteTTS", self.analyze_litetts)
        ]
        
        for name, analyzer_func in implementations:
            try:
                logger.info(f"Analyzing {name}...")
                result = analyzer_func()
                analysis_results["repositories_analyzed"].append(result)
                
                # Extract key findings
                if result.get("speecht5_usage", {}).get("speecht5_mentions"):
                    analysis_results["key_findings"].append(f"{name} uses SpeechT5 explicitly")
                
                if result.get("structure", {}).get("key_files", {}).get("model"):
                    analysis_results["key_findings"].append(f"{name} has dedicated model files")
                
            except Exception as e:
                logger.error(f"Failed to analyze {name}: {e}")
                analysis_results["repositories_analyzed"].append({
                    "repository": name,
                    "error": str(e)
                })
        
        # Analyze eSpeak integration
        espeak_analysis = self.analyze_espeak_integration()
        analysis_results["espeak_integration"] = espeak_analysis
        
        # Compare with JabberTTS
        comparison = self.compare_with_jabbertts()
        analysis_results["jabbertts_comparison"] = comparison
        
        # Extract critical differences
        analysis_results["critical_differences"] = comparison.get("differences", [])
        analysis_results["recommendations"] = comparison.get("recommendations", [])
        
        # Log summary
        logger.info("📊 ANALYSIS SUMMARY:")
        logger.info(f"Repositories analyzed: {len(analysis_results['repositories_analyzed'])}")
        logger.info(f"Key findings: {len(analysis_results['key_findings'])}")
        logger.info(f"Critical differences: {len(analysis_results['critical_differences'])}")
        
        for finding in analysis_results["key_findings"]:
            logger.info(f"  ✓ {finding}")
        
        for diff in analysis_results["critical_differences"]:
            logger.warning(f"  ⚠ {diff}")
        
        return analysis_results
    
    def save_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Save analysis report to file."""
        output_file = Path("temp") / "reference_implementation_analysis.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"📁 Analysis report saved to: {output_file}")
        return str(output_file)


def main():
    """Main analysis execution."""
    try:
        analyzer = ReferenceImplementationAnalyzer()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        # Save report
        report_file = analyzer.save_analysis_report(results)
        
        logger.info("✅ Reference implementation analysis completed")
        logger.info(f"📁 Detailed report: {report_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
