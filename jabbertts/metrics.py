"""JabberTTS Metrics Collection System.

This module provides real-time metrics collection for monitoring system performance,
request statistics, and TTS generation metrics.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Individual request metric data."""
    timestamp: float
    duration: float
    success: bool
    rtf: Optional[float] = None
    audio_duration: Optional[float] = None
    text_length: Optional[int] = None
    voice: Optional[str] = None
    format: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    uptime_seconds: float
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Real-time metrics collection and aggregation."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of request metrics to keep in memory
        """
        self.max_history = max_history
        self.start_time = time.time()
        
        # Request metrics storage
        self.request_history: deque = deque(maxlen=max_history)
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # System metrics
        self.system_metrics_history: deque = deque(maxlen=100)  # Keep last 100 system snapshots
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background system monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("Metrics collector initialized")
    
    def start_monitoring(self, interval: float = 30.0):
        """Start background system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped system monitoring")
    
    def _monitor_system(self, interval: float):
        """Background system monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self.system_metrics_history.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage (1-second average)
        cpu_percent = psutil.cpu_percent(interval=1.0)

        # Process-specific memory usage
        try:
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)  # Resident Set Size in MB

            # System memory for context
            system_memory = psutil.virtual_memory()
            system_memory_total_mb = system_memory.total / (1024 * 1024)

            # Calculate process memory percentage of total system memory
            process_memory_percent = (process_memory_mb / system_memory_total_mb) * 100

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback to system memory if process info unavailable
            system_memory = psutil.virtual_memory()
            process_memory_mb = system_memory.used / (1024 * 1024)
            system_memory_total_mb = system_memory.total / (1024 * 1024)
            process_memory_percent = system_memory.percent

        # Uptime
        uptime_seconds = time.time() - self.start_time

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=process_memory_percent,
            memory_used_mb=process_memory_mb,
            memory_total_mb=system_memory_total_mb,
            uptime_seconds=uptime_seconds
        )
    
    def record_request(self, 
                      duration: float,
                      success: bool,
                      rtf: Optional[float] = None,
                      audio_duration: Optional[float] = None,
                      text_length: Optional[int] = None,
                      voice: Optional[str] = None,
                      format: Optional[str] = None):
        """Record a TTS request metric.
        
        Args:
            duration: Request duration in seconds
            success: Whether the request was successful
            rtf: Real-time factor (optional)
            audio_duration: Generated audio duration in seconds (optional)
            text_length: Input text length (optional)
            voice: Voice used (optional)
            format: Audio format (optional)
        """
        metric = RequestMetric(
            timestamp=time.time(),
            duration=duration,
            success=success,
            rtf=rtf,
            audio_duration=audio_duration,
            text_length=text_length,
            voice=voice,
            format=format
        )
        
        with self._lock:
            self.request_history.append(metric)
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
    
    def get_performance_metrics(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get aggregated performance metrics.
        
        Args:
            window_minutes: Time window for recent metrics calculation
            
        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            current_time = time.time()
            window_start = current_time - (window_minutes * 60)
            
            # Filter recent requests
            recent_requests = [
                req for req in self.request_history 
                if req.timestamp >= window_start
            ]
            
            # Calculate metrics
            total_recent = len(recent_requests)
            successful_recent = sum(1 for req in recent_requests if req.success)
            failed_recent = total_recent - successful_recent
            
            # Average response time
            if recent_requests:
                avg_response_time = sum(req.duration for req in recent_requests) / total_recent
                
                # Average RTF for successful requests with RTF data
                rtf_values = [req.rtf for req in recent_requests if req.success and req.rtf is not None]
                avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0
                
                # Requests per minute
                requests_per_minute = total_recent / window_minutes if window_minutes > 0 else 0
            else:
                avg_response_time = 0.0
                avg_rtf = 0.0
                requests_per_minute = 0.0
            
            # Error rate
            error_rate = (failed_recent / total_recent * 100) if total_recent > 0 else 0.0
            
            # System metrics (latest)
            latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            # Uptime
            uptime_seconds = current_time - self.start_time
            uptime_hours = int(uptime_seconds // 3600)
            uptime_minutes = int((uptime_seconds % 3600) // 60)
            uptime_str = f"{uptime_hours}h {uptime_minutes}m"
            
            return {
                "rtf": round(avg_rtf, 3),
                "avg_response_time": round(avg_response_time, 2),
                "requests_per_minute": round(requests_per_minute, 1),
                "memory_usage": f"{latest_system.memory_used_mb:.1f}MB" if latest_system else "N/A",
                "memory_percent": f"{latest_system.memory_percent:.1f}%" if latest_system else "N/A",
                "cpu_usage": f"{latest_system.cpu_percent:.1f}%" if latest_system else "N/A",
                "uptime": uptime_str,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "error_rate": f"{error_rate:.1f}%",
                "recent_requests": total_recent,
                "recent_successful": successful_recent,
                "recent_failed": failed_recent,
                "window_minutes": window_minutes
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status.

        Returns:
            Dictionary with system status information
        """
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else None

        if latest_system:
            # Determine status based on resource usage
            if latest_system.cpu_percent > 90 or latest_system.memory_percent > 90:
                status = "critical"
            elif latest_system.cpu_percent > 70 or latest_system.memory_percent > 70:
                status = "warning"
            else:
                status = "healthy"
        else:
            status = "unknown"

        return {
            "status": status,
            "cpu_percent": latest_system.cpu_percent if latest_system else 0.0,
            "memory_percent": latest_system.memory_percent if latest_system else 0.0,
            "memory_used_mb": latest_system.memory_used_mb if latest_system else 0.0,
            "memory_total_mb": latest_system.memory_total_mb if latest_system else 0.0,
            "uptime_seconds": latest_system.uptime_seconds if latest_system else 0.0,
            "monitoring_active": self._monitoring
        }

    def validate_metrics_consistency(self) -> Dict[str, Any]:
        """Validate consistency of metrics across the system.

        Returns:
            Dictionary with validation results and any inconsistencies found
        """
        validation_results = {
            "consistent": True,
            "issues": [],
            "recommendations": []
        }

        with self._lock:
            if not self.request_history:
                validation_results["issues"].append("No request history available for validation")
                return validation_results

            recent_requests = list(self.request_history)[-10:]  # Last 10 requests

            for i, request in enumerate(recent_requests):
                if request.rtf is not None and request.audio_duration is not None:
                    # Validate RTF calculation
                    expected_rtf = request.duration / request.audio_duration if request.audio_duration > 0 else 0
                    rtf_diff = abs(request.rtf - expected_rtf)

                    if rtf_diff > 0.01:  # Allow small floating point differences
                        validation_results["consistent"] = False
                        validation_results["issues"].append(
                            f"Request {i}: RTF inconsistency - recorded: {request.rtf:.3f}, "
                            f"calculated: {expected_rtf:.3f}, diff: {rtf_diff:.3f}"
                        )

                # Validate audio duration is positive
                if request.audio_duration is not None and request.audio_duration <= 0:
                    validation_results["consistent"] = False
                    validation_results["issues"].append(
                        f"Request {i}: Invalid audio duration: {request.audio_duration}"
                    )

                # Validate inference time is positive
                if request.duration <= 0:
                    validation_results["consistent"] = False
                    validation_results["issues"].append(
                        f"Request {i}: Invalid inference time: {request.duration}"
                    )

        # Add recommendations based on issues found
        if not validation_results["consistent"]:
            validation_results["recommendations"].extend([
                "Check RTF calculation in inference engine",
                "Verify audio duration calculation after processing",
                "Ensure consistent timing measurements across components"
            ])

        return validation_results


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        _metrics_collector.start_monitoring()
    return _metrics_collector


def shutdown_metrics():
    """Shutdown the metrics collector."""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.stop_monitoring()
        _metrics_collector = None
