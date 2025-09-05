"""Self-Debugging Capabilities for TTS Validation.

This module provides automatic detection of common TTS issues, root cause analysis
for quality degradation, performance regression detection, and automated issue reporting.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from jabbertts.validation.metrics import get_validation_metrics
from jabbertts.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of TTS issues that can be detected."""
    PRONUNCIATION_ERROR = "pronunciation_error"
    AUDIO_QUALITY_DEGRADATION = "audio_quality_degradation"
    PERFORMANCE_REGRESSION = "performance_regression"
    ROBOTIC_SPEECH = "robotic_speech"
    SILENCE_ISSUES = "silence_issues"
    VOICE_INCONSISTENCY = "voice_inconsistency"
    FORMAT_ISSUES = "format_issues"
    SYSTEM_OVERLOAD = "system_overload"


class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DetectedIssue:
    """Represents a detected TTS issue."""
    issue_type: IssueType
    severity: IssueSeverity
    title: str
    description: str
    affected_components: List[str]
    root_cause: Optional[str] = None
    recommended_actions: List[str] = None
    evidence: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.recommended_actions is None:
            self.recommended_actions = []
        if self.evidence is None:
            self.evidence = {}


class SelfDebugger:
    """Automated self-debugging system for TTS validation."""
    
    def __init__(self):
        """Initialize self-debugger."""
        self.validation_metrics = get_validation_metrics()
        self.system_metrics = get_metrics_collector()
        self.issue_detectors = self._initialize_detectors()
        
        logger.info("Self-debugger initialized")
    
    def _initialize_detectors(self) -> Dict[str, callable]:
        """Initialize issue detection methods.
        
        Returns:
            Dictionary of detector methods
        """
        return {
            "pronunciation_errors": self._detect_pronunciation_errors,
            "audio_quality_degradation": self._detect_audio_quality_degradation,
            "performance_regression": self._detect_performance_regression,
            "robotic_speech": self._detect_robotic_speech,
            "silence_issues": self._detect_silence_issues,
            "voice_inconsistency": self._detect_voice_inconsistency,
            "format_issues": self._detect_format_issues,
            "system_overload": self._detect_system_overload
        }
    
    def run_full_diagnosis(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Run comprehensive system diagnosis.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            Complete diagnosis results
        """
        logger.info(f"Running full system diagnosis (window: {window_minutes} minutes)")
        
        start_time = time.time()
        detected_issues = []
        
        # Run all detectors
        for detector_name, detector_method in self.issue_detectors.items():
            try:
                logger.debug(f"Running detector: {detector_name}")
                issues = detector_method(window_minutes)
                detected_issues.extend(issues)
            except Exception as e:
                logger.error(f"Detector {detector_name} failed: {e}")
                # Create an issue for the failed detector
                detected_issues.append(DetectedIssue(
                    issue_type=IssueType.SYSTEM_OVERLOAD,
                    severity=IssueSeverity.MEDIUM,
                    title=f"Detector Failure: {detector_name}",
                    description=f"Issue detector {detector_name} failed to execute",
                    affected_components=["validation_system"],
                    root_cause=str(e),
                    recommended_actions=["Check system logs", "Restart validation system"]
                ))
        
        # Prioritize issues by severity
        critical_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.CRITICAL]
        high_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.HIGH]
        medium_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.MEDIUM]
        low_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.LOW]
        info_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.INFO]
        
        # Generate system health assessment
        health_assessment = self._assess_system_health(detected_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detected_issues)
        
        diagnosis_time = time.time() - start_time
        
        return {
            "diagnosis_timestamp": start_time,
            "diagnosis_duration": diagnosis_time,
            "window_minutes": window_minutes,
            "total_issues": len(detected_issues),
            "issues_by_severity": {
                "critical": len(critical_issues),
                "high": len(high_issues),
                "medium": len(medium_issues),
                "low": len(low_issues),
                "info": len(info_issues)
            },
            "detected_issues": [self._issue_to_dict(issue) for issue in detected_issues],
            "health_assessment": health_assessment,
            "recommendations": recommendations,
            "system_status": self._determine_system_status(detected_issues)
        }
    
    def _detect_pronunciation_errors(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect pronunciation accuracy issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected pronunciation issues
        """
        issues = []
        
        # Get validation summary
        summary = self.validation_metrics.get_validation_summary(window_minutes)
        category_breakdown = self.validation_metrics.get_category_breakdown(window_minutes)
        
        # Check overall accuracy
        avg_accuracy = summary.get("avg_accuracy", 0)
        if avg_accuracy < 0.7:
            severity = IssueSeverity.CRITICAL if avg_accuracy < 0.5 else IssueSeverity.HIGH
            issues.append(DetectedIssue(
                issue_type=IssueType.PRONUNCIATION_ERROR,
                severity=severity,
                title="Low Pronunciation Accuracy",
                description=f"Average pronunciation accuracy is {avg_accuracy:.2%}, below acceptable threshold",
                affected_components=["tts_engine", "audio_processor"],
                root_cause="Model quality degradation or input text complexity",
                recommended_actions=[
                    "Review recent model changes",
                    "Analyze failed test cases",
                    "Check input text complexity",
                    "Consider model retraining"
                ],
                evidence={"avg_accuracy": avg_accuracy, "threshold": 0.7}
            ))
        
        # Check category-specific issues
        for category, data in category_breakdown.items():
            if data.get("avg_accuracy", 0) < 0.6:
                issues.append(DetectedIssue(
                    issue_type=IssueType.PRONUNCIATION_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    title=f"Poor {category.title()} Pronunciation",
                    description=f"Pronunciation accuracy for {category} category is {data['avg_accuracy']:.2%}",
                    affected_components=["tts_engine"],
                    root_cause=f"Model struggles with {category} content",
                    recommended_actions=[
                        f"Review {category} training data",
                        f"Add more {category} examples to training set",
                        "Consider specialized preprocessing"
                    ],
                    evidence={"category": category, "accuracy": data["avg_accuracy"]}
                ))
        
        return issues
    
    def _detect_audio_quality_degradation(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect audio quality degradation issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected audio quality issues
        """
        issues = []
        
        summary = self.validation_metrics.get_validation_summary(window_minutes)
        avg_quality = summary.get("avg_quality_score", 0)
        
        if avg_quality < 0.6:
            severity = IssueSeverity.CRITICAL if avg_quality < 0.4 else IssueSeverity.HIGH
            issues.append(DetectedIssue(
                issue_type=IssueType.AUDIO_QUALITY_DEGRADATION,
                severity=severity,
                title="Audio Quality Degradation",
                description=f"Average audio quality score is {avg_quality:.2f}, below acceptable threshold",
                affected_components=["audio_processor", "tts_engine"],
                root_cause="Audio processing pipeline issues or model degradation",
                recommended_actions=[
                    "Check audio processing settings",
                    "Verify model integrity",
                    "Review recent audio pipeline changes",
                    "Test with known good audio samples"
                ],
                evidence={"avg_quality_score": avg_quality, "threshold": 0.6}
            ))
        
        return issues
    
    def _detect_performance_regression(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect performance regression issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected performance issues
        """
        issues = []
        
        # Check validation performance regression
        regression = self.validation_metrics.get_performance_regression()
        if regression.get("regression_detected", False):
            severity_map = {"high": IssueSeverity.CRITICAL, "medium": IssueSeverity.HIGH, "low": IssueSeverity.MEDIUM}
            severity = severity_map.get(regression.get("severity", "low"), IssueSeverity.MEDIUM)
            
            issues.append(DetectedIssue(
                issue_type=IssueType.PERFORMANCE_REGRESSION,
                severity=severity,
                title="Performance Regression Detected",
                description=f"Quality regression detected: {regression.get('severity', 'unknown')} severity",
                affected_components=["tts_engine", "validation_system"],
                root_cause="Recent changes affecting system performance",
                recommended_actions=[
                    "Review recent code changes",
                    "Check system resource usage",
                    "Rollback recent updates if necessary",
                    "Investigate performance bottlenecks"
                ],
                evidence=regression
            ))
        
        # Check system performance metrics
        system_metrics = self.system_metrics.get_performance_metrics()
        avg_rtf = system_metrics.get("rtf", 0)
        
        if avg_rtf > 2.0:
            severity = IssueSeverity.HIGH if avg_rtf > 3.0 else IssueSeverity.MEDIUM
            issues.append(DetectedIssue(
                issue_type=IssueType.PERFORMANCE_REGRESSION,
                severity=severity,
                title="High Real-Time Factor",
                description=f"Average RTF is {avg_rtf:.2f}, exceeding target of 2.0",
                affected_components=["tts_engine", "inference_pipeline"],
                root_cause="Inference pipeline performance degradation",
                recommended_actions=[
                    "Optimize inference pipeline",
                    "Check system resource availability",
                    "Consider model optimization",
                    "Review concurrent request handling"
                ],
                evidence={"avg_rtf": avg_rtf, "target": 2.0}
            ))
        
        return issues
    
    def _detect_robotic_speech(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect robotic speech patterns.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected robotic speech issues
        """
        issues = []
        
        summary = self.validation_metrics.get_validation_summary(window_minutes)
        avg_quality = summary.get("avg_quality_score", 0)
        
        # Robotic speech typically shows up as low naturalness scores
        if avg_quality < 0.7 and summary.get("total_validations", 0) > 5:
            issues.append(DetectedIssue(
                issue_type=IssueType.ROBOTIC_SPEECH,
                severity=IssueSeverity.MEDIUM,
                title="Potential Robotic Speech Patterns",
                description="Quality scores suggest unnatural speech characteristics",
                affected_components=["tts_engine", "audio_processor"],
                root_cause="Model producing mechanical-sounding speech",
                recommended_actions=[
                    "Review prosody settings",
                    "Check audio enhancement parameters",
                    "Analyze speech naturalness metrics",
                    "Consider model fine-tuning"
                ],
                evidence={"avg_quality_score": avg_quality}
            ))
        
        return issues
    
    def _detect_silence_issues(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect silence and timing issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected silence issues
        """
        issues = []
        
        # This would require more detailed audio analysis
        # For now, we'll use validation failure patterns as a proxy
        failure_analysis = self.validation_metrics.get_failure_analysis(window_minutes)
        
        if failure_analysis.get("total_failures", 0) > 5:
            common_errors = failure_analysis.get("common_errors", [])
            silence_related = [error for error in common_errors if "silence" in error.get("error", "").lower()]
            
            if silence_related:
                issues.append(DetectedIssue(
                    issue_type=IssueType.SILENCE_ISSUES,
                    severity=IssueSeverity.MEDIUM,
                    title="Silence-Related Validation Failures",
                    description="Multiple validation failures related to silence detection",
                    affected_components=["audio_processor", "validation_system"],
                    root_cause="Audio silence detection or processing issues",
                    recommended_actions=[
                        "Check audio preprocessing pipeline",
                        "Review silence detection thresholds",
                        "Verify audio format handling"
                    ],
                    evidence={"silence_errors": silence_related}
                ))
        
        return issues
    
    def _detect_voice_inconsistency(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect voice consistency issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected voice consistency issues
        """
        issues = []
        
        voice_breakdown = self.validation_metrics.get_voice_breakdown(window_minutes)
        
        # Check for significant performance differences between voices
        voice_scores = {voice: data.get("avg_quality_score", 0) for voice, data in voice_breakdown.items()}
        
        if len(voice_scores) > 1:
            max_score = max(voice_scores.values())
            min_score = min(voice_scores.values())
            
            if max_score - min_score > 0.3:  # 30% difference
                worst_voice = min(voice_scores, key=voice_scores.get)
                issues.append(DetectedIssue(
                    issue_type=IssueType.VOICE_INCONSISTENCY,
                    severity=IssueSeverity.MEDIUM,
                    title="Voice Quality Inconsistency",
                    description=f"Significant quality difference between voices (range: {min_score:.2f} - {max_score:.2f})",
                    affected_components=["tts_engine", "voice_models"],
                    root_cause=f"Voice '{worst_voice}' performing significantly worse than others",
                    recommended_actions=[
                        f"Review {worst_voice} voice model",
                        "Check voice-specific configurations",
                        "Consider voice model retraining",
                        "Verify voice model integrity"
                    ],
                    evidence={"voice_scores": voice_scores, "worst_voice": worst_voice}
                ))
        
        return issues
    
    def _detect_format_issues(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect audio format-related issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected format issues
        """
        issues = []
        
        failure_analysis = self.validation_metrics.get_failure_analysis(window_minutes)
        common_errors = failure_analysis.get("common_errors", [])
        
        format_errors = [error for error in common_errors if any(fmt in error.get("error", "").lower() for fmt in ["format", "codec", "encoding"])]
        
        if format_errors:
            issues.append(DetectedIssue(
                issue_type=IssueType.FORMAT_ISSUES,
                severity=IssueSeverity.MEDIUM,
                title="Audio Format Processing Issues",
                description="Multiple failures related to audio format processing",
                affected_components=["audio_processor", "format_handlers"],
                root_cause="Audio format encoding or decoding issues",
                recommended_actions=[
                    "Check audio format support",
                    "Verify codec availability",
                    "Review format conversion pipeline",
                    "Test with different audio formats"
                ],
                evidence={"format_errors": format_errors}
            ))
        
        return issues
    
    def _detect_system_overload(self, window_minutes: int) -> List[DetectedIssue]:
        """Detect system overload issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            List of detected system overload issues
        """
        issues = []
        
        system_status = self.system_metrics.get_system_status()
        cpu_percent = system_status.get("cpu_percent", 0)
        memory_percent = system_status.get("memory_percent", 0)
        
        if cpu_percent > 90:
            issues.append(DetectedIssue(
                issue_type=IssueType.SYSTEM_OVERLOAD,
                severity=IssueSeverity.HIGH,
                title="High CPU Usage",
                description=f"CPU usage is {cpu_percent:.1f}%, indicating system overload",
                affected_components=["system", "inference_pipeline"],
                root_cause="High computational load or inefficient processing",
                recommended_actions=[
                    "Reduce concurrent requests",
                    "Optimize inference pipeline",
                    "Scale system resources",
                    "Check for resource leaks"
                ],
                evidence={"cpu_percent": cpu_percent}
            ))
        
        if memory_percent > 90:
            issues.append(DetectedIssue(
                issue_type=IssueType.SYSTEM_OVERLOAD,
                severity=IssueSeverity.HIGH,
                title="High Memory Usage",
                description=f"Memory usage is {memory_percent:.1f}%, indicating potential memory issues",
                affected_components=["system", "model_manager"],
                root_cause="Memory leaks or excessive memory allocation",
                recommended_actions=[
                    "Check for memory leaks",
                    "Optimize model loading",
                    "Implement memory cleanup",
                    "Scale memory resources"
                ],
                evidence={"memory_percent": memory_percent}
            ))
        
        return issues
    
    def _assess_system_health(self, detected_issues: List[DetectedIssue]) -> Dict[str, Any]:
        """Assess overall system health based on detected issues.
        
        Args:
            detected_issues: List of detected issues
            
        Returns:
            System health assessment
        """
        if not detected_issues:
            return {
                "status": "healthy",
                "score": 100,
                "description": "No issues detected"
            }
        
        # Calculate health score based on issue severity
        score = 100
        for issue in detected_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                score -= 25
            elif issue.severity == IssueSeverity.HIGH:
                score -= 15
            elif issue.severity == IssueSeverity.MEDIUM:
                score -= 10
            elif issue.severity == IssueSeverity.LOW:
                score -= 5
            # INFO issues don't affect score
        
        score = max(0, score)  # Don't go below 0
        
        # Determine status
        if score >= 90:
            status = "healthy"
        elif score >= 70:
            status = "warning"
        elif score >= 50:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            "status": status,
            "score": score,
            "description": f"System health score: {score}/100"
        }
    
    def _generate_recommendations(self, detected_issues: List[DetectedIssue]) -> List[str]:
        """Generate prioritized recommendations based on detected issues.
        
        Args:
            detected_issues: List of detected issues
            
        Returns:
            List of prioritized recommendations
        """
        if not detected_issues:
            return ["System is operating normally"]
        
        # Collect all recommendations and prioritize by severity
        recommendations = []
        
        critical_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.CRITICAL]
        high_issues = [issue for issue in detected_issues if issue.severity == IssueSeverity.HIGH]
        
        if critical_issues:
            recommendations.append("URGENT: Address critical issues immediately")
            for issue in critical_issues[:3]:  # Top 3 critical issues
                recommendations.extend(issue.recommended_actions[:2])  # Top 2 actions per issue
        
        if high_issues:
            recommendations.append("HIGH PRIORITY: Address high-severity issues")
            for issue in high_issues[:2]:  # Top 2 high issues
                recommendations.extend(issue.recommended_actions[:1])  # Top action per issue
        
        # Add general recommendations
        if len(detected_issues) > 5:
            recommendations.append("Consider comprehensive system review due to multiple issues")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _determine_system_status(self, detected_issues: List[DetectedIssue]) -> str:
        """Determine overall system status.
        
        Args:
            detected_issues: List of detected issues
            
        Returns:
            System status string
        """
        if not detected_issues:
            return "operational"
        
        critical_count = sum(1 for issue in detected_issues if issue.severity == IssueSeverity.CRITICAL)
        high_count = sum(1 for issue in detected_issues if issue.severity == IssueSeverity.HIGH)
        
        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "degraded"
        elif high_count > 0:
            return "warning"
        else:
            return "operational"
    
    def _issue_to_dict(self, issue: DetectedIssue) -> Dict[str, Any]:
        """Convert DetectedIssue to dictionary.
        
        Args:
            issue: DetectedIssue instance
            
        Returns:
            Issue as dictionary
        """
        return {
            "issue_type": issue.issue_type.value,
            "severity": issue.severity.value,
            "title": issue.title,
            "description": issue.description,
            "affected_components": issue.affected_components,
            "root_cause": issue.root_cause,
            "recommended_actions": issue.recommended_actions,
            "evidence": issue.evidence,
            "timestamp": issue.timestamp
        }
    
    def get_issue_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get a quick summary of current issues.
        
        Args:
            window_minutes: Time window for analysis
            
        Returns:
            Issue summary
        """
        diagnosis = self.run_full_diagnosis(window_minutes)
        
        return {
            "system_status": diagnosis["system_status"],
            "health_score": diagnosis["health_assessment"]["score"],
            "total_issues": diagnosis["total_issues"],
            "critical_issues": diagnosis["issues_by_severity"]["critical"],
            "high_issues": diagnosis["issues_by_severity"]["high"],
            "top_recommendations": diagnosis["recommendations"][:3]
        }


# Global self-debugger instance
_self_debugger: Optional[SelfDebugger] = None


def get_self_debugger() -> SelfDebugger:
    """Get the global self-debugger instance.
    
    Returns:
        Global SelfDebugger instance
    """
    global _self_debugger
    if _self_debugger is None:
        _self_debugger = SelfDebugger()
    return _self_debugger
