"""Validation Metrics Collection and Analysis.

This module provides metrics collection specifically for validation
and quality assessment tracking.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetric:
    """Individual validation metric data."""
    timestamp: float
    test_category: str
    voice: str
    format: str
    speed: float
    success: bool
    accuracy_score: float
    quality_score: float
    rtf: Optional[float] = None
    inference_time: Optional[float] = None
    validation_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class QualityTrend:
    """Quality trend analysis data."""
    timestamp: float
    avg_accuracy: float
    avg_quality_score: float
    success_rate: float
    avg_rtf: float
    sample_count: int


class ValidationMetrics:
    """Validation metrics collection and analysis."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize validation metrics collector.
        
        Args:
            max_history: Maximum number of validation metrics to keep
        """
        self.max_history = max_history
        self.start_time = time.time()
        
        # Validation metrics storage
        self.validation_history: deque = deque(maxlen=max_history)
        self.quality_trends: deque = deque(maxlen=1000)  # Keep last 1000 trend points
        
        # Counters
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("Validation metrics collector initialized")
    
    def record_validation(self,
                         test_category: str,
                         voice: str,
                         format: str,
                         speed: float,
                         success: bool,
                         accuracy_score: float,
                         quality_score: float,
                         rtf: Optional[float] = None,
                         inference_time: Optional[float] = None,
                         validation_time: Optional[float] = None,
                         error_message: Optional[str] = None):
        """Record a validation metric.
        
        Args:
            test_category: Category of the test
            voice: Voice used
            format: Audio format
            speed: Speech speed
            success: Whether validation was successful
            accuracy_score: Accuracy score (0.0-1.0)
            quality_score: Quality score (0.0-1.0)
            rtf: Real-time factor (optional)
            inference_time: TTS inference time (optional)
            validation_time: Validation processing time (optional)
            error_message: Error message if failed (optional)
        """
        metric = ValidationMetric(
            timestamp=time.time(),
            test_category=test_category,
            voice=voice,
            format=format,
            speed=speed,
            success=success,
            accuracy_score=accuracy_score,
            quality_score=quality_score,
            rtf=rtf,
            inference_time=inference_time,
            validation_time=validation_time,
            error_message=error_message
        )
        
        with self._lock:
            self.validation_history.append(metric)
            self.total_validations += 1
            if success:
                self.successful_validations += 1
            else:
                self.failed_validations += 1
    
    def get_validation_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get validation summary for a time window.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Validation summary
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)
            
            # Filter recent validations
            recent_validations = [
                val for val in self.validation_history
                if val.timestamp >= window_start
            ]
            
            if not recent_validations:
                return {
                    "window_minutes": window_minutes,
                    "total_validations": 0,
                    "success_rate": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_quality_score": 0.0,
                    "avg_rtf": 0.0,
                    "avg_inference_time": 0.0,
                    "avg_validation_time": 0.0
                }
            
            # Calculate metrics
            successful = [val for val in recent_validations if val.success]
            
            success_rate = len(successful) / len(recent_validations)
            avg_accuracy = sum(val.accuracy_score for val in successful) / len(successful) if successful else 0.0
            avg_quality_score = sum(val.quality_score for val in successful) / len(successful) if successful else 0.0
            
            # RTF and timing metrics
            rtf_values = [val.rtf for val in successful if val.rtf is not None]
            inference_times = [val.inference_time for val in successful if val.inference_time is not None]
            validation_times = [val.validation_time for val in successful if val.validation_time is not None]
            
            avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
            avg_validation_time = sum(validation_times) / len(validation_times) if validation_times else 0.0
            
            return {
                "window_minutes": window_minutes,
                "total_validations": len(recent_validations),
                "successful_validations": len(successful),
                "failed_validations": len(recent_validations) - len(successful),
                "success_rate": success_rate,
                "avg_accuracy": avg_accuracy,
                "avg_quality_score": avg_quality_score,
                "avg_rtf": avg_rtf,
                "avg_inference_time": avg_inference_time,
                "avg_validation_time": avg_validation_time,
                "quality_grade": self._score_to_grade(avg_quality_score),
                "accuracy_grade": self._score_to_grade(avg_accuracy)
            }
    
    def get_category_breakdown(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get validation breakdown by category.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Category breakdown
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)
            
            recent_validations = [
                val for val in self.validation_history
                if val.timestamp >= window_start
            ]
            
            categories = {}
            for val in recent_validations:
                category = val.test_category
                if category not in categories:
                    categories[category] = {
                        "total": 0,
                        "successful": 0,
                        "accuracies": [],
                        "quality_scores": []
                    }
                
                cat_data = categories[category]
                cat_data["total"] += 1
                if val.success:
                    cat_data["successful"] += 1
                    cat_data["accuracies"].append(val.accuracy_score)
                    cat_data["quality_scores"].append(val.quality_score)
            
            # Calculate averages
            for category, data in categories.items():
                data["success_rate"] = data["successful"] / data["total"] if data["total"] > 0 else 0.0
                data["avg_accuracy"] = sum(data["accuracies"]) / len(data["accuracies"]) if data["accuracies"] else 0.0
                data["avg_quality_score"] = sum(data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0.0
                
                # Clean up temporary lists
                del data["accuracies"]
                del data["quality_scores"]
            
            return categories
    
    def get_voice_breakdown(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get validation breakdown by voice.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Voice breakdown
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)
            
            recent_validations = [
                val for val in self.validation_history
                if val.timestamp >= window_start
            ]
            
            voices = {}
            for val in recent_validations:
                voice = val.voice
                if voice not in voices:
                    voices[voice] = {
                        "total": 0,
                        "successful": 0,
                        "accuracies": [],
                        "quality_scores": []
                    }
                
                voice_data = voices[voice]
                voice_data["total"] += 1
                if val.success:
                    voice_data["successful"] += 1
                    voice_data["accuracies"].append(val.accuracy_score)
                    voice_data["quality_scores"].append(val.quality_score)
            
            # Calculate averages
            for voice, data in voices.items():
                data["success_rate"] = data["successful"] / data["total"] if data["total"] > 0 else 0.0
                data["avg_accuracy"] = sum(data["accuracies"]) / len(data["accuracies"]) if data["accuracies"] else 0.0
                data["avg_quality_score"] = sum(data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0.0
                
                # Clean up temporary lists
                del data["accuracies"]
                del data["quality_scores"]
            
            return voices
    
    def get_quality_trends(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get quality trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            List of quality trend data points
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - (hours * 3600)
            
            # Group validations by hour
            hourly_data = {}
            for val in self.validation_history:
                if val.timestamp < window_start:
                    continue
                
                hour_key = int(val.timestamp // 3600) * 3600  # Round to hour
                if hour_key not in hourly_data:
                    hourly_data[hour_key] = {
                        "validations": [],
                        "successful": []
                    }
                
                hourly_data[hour_key]["validations"].append(val)
                if val.success:
                    hourly_data[hour_key]["successful"].append(val)
            
            # Calculate trends
            trends = []
            for hour_timestamp in sorted(hourly_data.keys()):
                data = hourly_data[hour_timestamp]
                successful = data["successful"]
                total = data["validations"]
                
                if total:
                    avg_accuracy = sum(val.accuracy_score for val in successful) / len(successful) if successful else 0.0
                    avg_quality_score = sum(val.quality_score for val in successful) / len(successful) if successful else 0.0
                    success_rate = len(successful) / len(total)
                    
                    rtf_values = [val.rtf for val in successful if val.rtf is not None]
                    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
                    
                    trends.append({
                        "timestamp": hour_timestamp,
                        "datetime": datetime.fromtimestamp(hour_timestamp).isoformat(),
                        "avg_accuracy": avg_accuracy,
                        "avg_quality_score": avg_quality_score,
                        "success_rate": success_rate,
                        "avg_rtf": avg_rtf,
                        "sample_count": len(total)
                    })
            
            return trends
    
    def get_failure_analysis(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze validation failures.
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Failure analysis
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)
            
            recent_failures = [
                val for val in self.validation_history
                if val.timestamp >= window_start and not val.success
            ]
            
            if not recent_failures:
                return {
                    "total_failures": 0,
                    "failure_categories": {},
                    "failure_voices": {},
                    "common_errors": []
                }
            
            # Analyze failure patterns
            failure_categories = {}
            failure_voices = {}
            error_messages = []
            
            for failure in recent_failures:
                # Category analysis
                category = failure.test_category
                failure_categories[category] = failure_categories.get(category, 0) + 1
                
                # Voice analysis
                voice = failure.voice
                failure_voices[voice] = failure_voices.get(voice, 0) + 1
                
                # Error message collection
                if failure.error_message:
                    error_messages.append(failure.error_message)
            
            # Find common error patterns
            common_errors = []
            error_counts = {}
            for error in error_messages:
                # Simplify error messages for pattern detection
                simplified_error = error.split(':')[0] if ':' in error else error
                error_counts[simplified_error] = error_counts.get(simplified_error, 0) + 1
            
            # Sort by frequency
            common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total_failures": len(recent_failures),
                "failure_categories": failure_categories,
                "failure_voices": failure_voices,
                "common_errors": [{"error": error, "count": count} for error, count in common_errors],
                "failure_rate": len(recent_failures) / (len(recent_failures) + self.successful_validations) if (len(recent_failures) + self.successful_validations) > 0 else 0.0
            }
    
    def get_performance_regression(self, baseline_hours: int = 24, comparison_hours: int = 1) -> Dict[str, Any]:
        """Detect performance regression by comparing recent performance to baseline.
        
        Args:
            baseline_hours: Hours to use for baseline calculation
            comparison_hours: Recent hours to compare against baseline
            
        Returns:
            Regression analysis
        """
        current_time = time.time()
        
        # Get baseline metrics
        baseline_start = current_time - (baseline_hours * 3600)
        baseline_end = current_time - (comparison_hours * 3600)
        
        comparison_start = current_time - (comparison_hours * 3600)
        
        with self._lock:
            baseline_validations = [
                val for val in self.validation_history
                if baseline_start <= val.timestamp <= baseline_end and val.success
            ]
            
            comparison_validations = [
                val for val in self.validation_history
                if val.timestamp >= comparison_start and val.success
            ]
        
        if not baseline_validations or not comparison_validations:
            return {
                "regression_detected": False,
                "insufficient_data": True
            }
        
        # Calculate baseline metrics
        baseline_accuracy = sum(val.accuracy_score for val in baseline_validations) / len(baseline_validations)
        baseline_quality = sum(val.quality_score for val in baseline_validations) / len(baseline_validations)
        
        # Calculate comparison metrics
        comparison_accuracy = sum(val.accuracy_score for val in comparison_validations) / len(comparison_validations)
        comparison_quality = sum(val.quality_score for val in comparison_validations) / len(comparison_validations)
        
        # Detect regression (>5% decrease)
        accuracy_regression = (baseline_accuracy - comparison_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
        quality_regression = (baseline_quality - comparison_quality) / baseline_quality if baseline_quality > 0 else 0
        
        regression_threshold = 0.05  # 5% decrease
        regression_detected = accuracy_regression > regression_threshold or quality_regression > regression_threshold
        
        return {
            "regression_detected": regression_detected,
            "baseline_period_hours": baseline_hours,
            "comparison_period_hours": comparison_hours,
            "baseline_accuracy": baseline_accuracy,
            "comparison_accuracy": comparison_accuracy,
            "accuracy_change": -accuracy_regression,  # Negative for decrease
            "baseline_quality": baseline_quality,
            "comparison_quality": comparison_quality,
            "quality_change": -quality_regression,  # Negative for decrease
            "baseline_sample_count": len(baseline_validations),
            "comparison_sample_count": len(comparison_validations),
            "severity": "high" if max(accuracy_regression, quality_regression) > 0.1 else "medium" if regression_detected else "low"
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade.
        
        Args:
            score: Numerical score (0.0-1.0)
            
        Returns:
            Letter grade
        """
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall validation system health.
        
        Returns:
            Overall health assessment
        """
        summary = self.get_validation_summary(window_minutes=60)
        failure_analysis = self.get_failure_analysis(window_minutes=60)
        regression = self.get_performance_regression()
        
        # Determine health status
        health_score = 0
        health_factors = []
        
        # Success rate factor (40% weight)
        success_rate = summary.get("success_rate", 0)
        if success_rate >= 0.95:
            health_score += 40
            health_factors.append("Excellent success rate")
        elif success_rate >= 0.85:
            health_score += 32
            health_factors.append("Good success rate")
        elif success_rate >= 0.75:
            health_score += 24
            health_factors.append("Fair success rate")
        else:
            health_score += 16
            health_factors.append("Poor success rate")
        
        # Quality factor (30% weight)
        avg_quality = summary.get("avg_quality_score", 0)
        if avg_quality >= 0.8:
            health_score += 30
            health_factors.append("High quality scores")
        elif avg_quality >= 0.7:
            health_score += 24
            health_factors.append("Good quality scores")
        elif avg_quality >= 0.6:
            health_score += 18
            health_factors.append("Fair quality scores")
        else:
            health_score += 12
            health_factors.append("Low quality scores")
        
        # Regression factor (20% weight)
        if not regression.get("regression_detected", False):
            health_score += 20
            health_factors.append("No performance regression")
        elif regression.get("severity", "low") == "low":
            health_score += 15
            health_factors.append("Minor performance regression")
        elif regression.get("severity", "low") == "medium":
            health_score += 10
            health_factors.append("Moderate performance regression")
        else:
            health_score += 5
            health_factors.append("Significant performance regression")
        
        # Activity factor (10% weight)
        total_validations = summary.get("total_validations", 0)
        if total_validations >= 10:
            health_score += 10
            health_factors.append("Sufficient validation activity")
        elif total_validations >= 5:
            health_score += 7
            health_factors.append("Moderate validation activity")
        else:
            health_score += 3
            health_factors.append("Low validation activity")
        
        # Determine overall health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 80:
            health_status = "good"
        elif health_score >= 70:
            health_status = "fair"
        elif health_score >= 60:
            health_status = "poor"
        else:
            health_status = "critical"
        
        return {
            "health_status": health_status,
            "health_score": health_score,
            "health_factors": health_factors,
            "summary": summary,
            "failure_analysis": failure_analysis,
            "regression_analysis": regression,
            "recommendations": self._generate_health_recommendations(health_status, summary, failure_analysis, regression)
        }
    
    def _generate_health_recommendations(self, health_status: str, summary: Dict, failure_analysis: Dict, regression: Dict) -> List[str]:
        """Generate health improvement recommendations.
        
        Args:
            health_status: Overall health status
            summary: Validation summary
            failure_analysis: Failure analysis
            regression: Regression analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if health_status in ["poor", "critical"]:
            recommendations.append("Immediate attention required for validation system")
        
        if summary.get("success_rate", 0) < 0.8:
            recommendations.append("Investigate and fix validation failures")
        
        if summary.get("avg_quality_score", 0) < 0.7:
            recommendations.append("Improve TTS quality to meet validation standards")
        
        if regression.get("regression_detected", False):
            recommendations.append("Address performance regression issues")
        
        if failure_analysis.get("total_failures", 0) > 5:
            recommendations.append("Analyze and resolve common failure patterns")
        
        if summary.get("total_validations", 0) < 5:
            recommendations.append("Increase validation testing frequency")
        
        if not recommendations:
            recommendations.append("Validation system is performing well")
        
        return recommendations


# Global validation metrics instance
_validation_metrics: Optional[ValidationMetrics] = None


def get_validation_metrics() -> ValidationMetrics:
    """Get the global validation metrics instance.
    
    Returns:
        Global ValidationMetrics instance
    """
    global _validation_metrics
    if _validation_metrics is None:
        _validation_metrics = ValidationMetrics()
    return _validation_metrics
