"""
Comprehensive metrics collection service for CUDA Evaluation Server V2
Tracks detailed performance metrics across all server operations
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompilationMetrics:
    """Metrics for compilation operations"""
    total_compilations: int = 0
    successful_compilations: int = 0
    failed_compilations: int = 0
    total_compilation_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    compilation_times: List[float] = field(default_factory=list)
    
    def record_compilation(self, success: bool, time_seconds: float, cache_hit: bool):
        """Record a compilation event"""
        self.total_compilations += 1
        self.total_compilation_time += time_seconds
        self.compilation_times.append(time_seconds)
        
        if success:
            self.successful_compilations += 1
        else:
            self.failed_compilations += 1
            
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compilation statistics"""
        if self.total_compilations == 0:
            return {
                "total_compilations": 0,
                "average_time_seconds": 0.0,
                "success_rate": 1.0,
                "cache_hit_rate": 0.0,
                "total_time_seconds": 0.0
            }
        
        return {
            "total_compilations": self.total_compilations,
            "successful_compilations": self.successful_compilations,
            "failed_compilations": self.failed_compilations,
            "total_time_seconds": round(self.total_compilation_time, 2),
            "average_time_seconds": round(self.total_compilation_time / self.total_compilations, 2),
            "success_rate": round(self.successful_compilations / self.total_compilations, 3),
            "cache_hit_rate": round(self.cache_hits / self.total_compilations, 3) if self.total_compilations > 0 else 0.0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "min_time_seconds": round(min(self.compilation_times), 2) if self.compilation_times else 0.0,
            "max_time_seconds": round(max(self.compilation_times), 2) if self.compilation_times else 0.0
        }


@dataclass
class ProfilingMetrics:
    """Metrics for profiling operations"""
    total_profilings: int = 0
    successful_profilings: int = 0
    failed_profilings: int = 0
    total_profiling_time: float = 0.0
    correctness_passed: int = 0
    correctness_failed: int = 0
    profiling_times: List[float] = field(default_factory=list)
    gpu_usage_times: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    
    def record_profiling(self, success: bool, time_seconds: float, gpu_id: Optional[int], 
                        correctness: Optional[bool]):
        """Record a profiling event"""
        self.total_profilings += 1
        self.total_profiling_time += time_seconds
        self.profiling_times.append(time_seconds)
        
        if success:
            self.successful_profilings += 1
        else:
            self.failed_profilings += 1
            
        if gpu_id is not None:
            self.gpu_usage_times[gpu_id] += time_seconds
            
        if correctness is True:
            self.correctness_passed += 1
        elif correctness is False:
            self.correctness_failed += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        if self.total_profilings == 0:
            return {
                "total_profilings": 0,
                "average_time_seconds": 0.0,
                "success_rate": 1.0,
                "correctness_rate": 1.0,
                "total_time_seconds": 0.0,
                "gpu_usage_times": {}
            }
        
        return {
            "total_profilings": self.total_profilings,
            "successful_profilings": self.successful_profilings,
            "failed_profilings": self.failed_profilings,
            "total_time_seconds": round(self.total_profiling_time, 2),
            "average_time_seconds": round(self.total_profiling_time / self.total_profilings, 2),
            "success_rate": round(self.successful_profilings / self.total_profilings, 3),
            "correctness_passed": self.correctness_passed,
            "correctness_failed": self.correctness_failed,
            "correctness_rate": round(self.correctness_passed / (self.correctness_passed + self.correctness_failed), 3) if (self.correctness_passed + self.correctness_failed) > 0 else 1.0,
            "min_time_seconds": round(min(self.profiling_times), 2) if self.profiling_times else 0.0,
            "max_time_seconds": round(max(self.profiling_times), 2) if self.profiling_times else 0.0,
            "gpu_usage_times": {gpu_id: round(time_sec, 2) for gpu_id, time_sec in self.gpu_usage_times.items()}
        }


@dataclass
class GPUUtilizationMetrics:
    """Metrics for GPU resource utilization"""
    gpu_acquisitions: int = 0
    gpu_releases: int = 0
    total_gpu_time: float = 0.0
    total_wait_time: float = 0.0
    wait_times: List[float] = field(default_factory=list)
    gpu_usage_by_id: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    active_gpu_sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # job_id -> {gpu_id, start_time}
    
    def record_gpu_acquisition(self, job_id: str, gpu_id: int, wait_time_seconds: float):
        """Record GPU acquisition"""
        self.gpu_acquisitions += 1
        self.total_wait_time += wait_time_seconds
        self.wait_times.append(wait_time_seconds)
        
        self.active_gpu_sessions[job_id] = {
            "gpu_id": gpu_id,
            "start_time": time.time(),
            "wait_time": wait_time_seconds
        }
    
    def record_gpu_release(self, job_id: str):
        """Record GPU release"""
        if job_id in self.active_gpu_sessions:
            session = self.active_gpu_sessions[job_id]
            gpu_id = session["gpu_id"]
            session_time = time.time() - session["start_time"]
            
            self.gpu_releases += 1
            self.total_gpu_time += session_time
            self.gpu_usage_by_id[gpu_id] += session_time
            
            del self.active_gpu_sessions[job_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GPU utilization statistics"""
        return {
            "gpu_acquisitions": self.gpu_acquisitions,
            "gpu_releases": self.gpu_releases,
            "active_sessions": len(self.active_gpu_sessions),
            "total_gpu_time_seconds": round(self.total_gpu_time, 2),
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "average_wait_time_seconds": round(self.total_wait_time / self.gpu_acquisitions, 2) if self.gpu_acquisitions > 0 else 0.0,
            "max_wait_time_seconds": round(max(self.wait_times), 2) if self.wait_times else 0.0,
            "min_wait_time_seconds": round(min(self.wait_times), 2) if self.wait_times else 0.0,
            "utilization_by_gpu": {gpu_id: round(time_sec, 2) for gpu_id, time_sec in self.gpu_usage_by_id.items()}
        }


@dataclass
class ThroughputMetrics:
    """Metrics for overall server throughput"""
    server_start_time: float = field(default_factory=time.time)
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    total_end_to_end_time: float = 0.0
    end_to_end_times: List[float] = field(default_factory=list)
    request_timestamps: List[float] = field(default_factory=list)
    
    def record_request_start(self):
        """Record the start of a request"""
        self.total_requests += 1
        self.request_timestamps.append(time.time())
    
    def record_request_completion(self, success: bool, end_to_end_time_seconds: float):
        """Record request completion"""
        if success:
            self.completed_requests += 1
        else:
            self.failed_requests += 1
            
        self.total_end_to_end_time += end_to_end_time_seconds
        self.end_to_end_times.append(end_to_end_time_seconds)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throughput statistics"""
        current_time = time.time()
        uptime_seconds = current_time - self.server_start_time
        uptime_minutes = uptime_seconds / 60.0
        
        # Calculate requests per minute
        requests_per_minute = (self.total_requests / uptime_minutes) if uptime_minutes > 0 else 0.0
        
        # Calculate recent throughput (last 5 minutes)
        recent_cutoff = current_time - 300  # 5 minutes ago
        recent_requests = sum(1 for ts in self.request_timestamps if ts >= recent_cutoff)
        recent_throughput = (recent_requests / 5.0) if recent_requests > 0 else 0.0
        
        return {
            "server_uptime_seconds": round(uptime_seconds, 2),
            "server_uptime_minutes": round(uptime_minutes, 2),
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.completed_requests / self.total_requests, 3) if self.total_requests > 0 else 1.0,
            "requests_per_minute": round(requests_per_minute, 2),
            "recent_throughput_per_minute": round(recent_throughput, 2),
            "average_end_to_end_time_seconds": round(self.total_end_to_end_time / len(self.end_to_end_times), 2) if self.end_to_end_times else 0.0,
            "min_end_to_end_time_seconds": round(min(self.end_to_end_times), 2) if self.end_to_end_times else 0.0,
            "max_end_to_end_time_seconds": round(max(self.end_to_end_times), 2) if self.end_to_end_times else 0.0
        }


class MetricsCollector:
    """
    Comprehensive metrics collection service
    Thread-safe collector for all server performance metrics
    """
    
    def __init__(self):
        self.compilation_metrics = CompilationMetrics()
        self.profiling_metrics = ProfilingMetrics()
        self.gpu_utilization_metrics = GPUUtilizationMetrics()
        self.throughput_metrics = ThroughputMetrics()
        self._lock = threading.RLock()
        
        logger.info("MetricsCollector initialized")
    
    # Compilation metrics
    def record_compilation_start(self, job_id: str):
        """Record compilation start (for tracking active compilations)"""
        # Could be used for tracking concurrent compilations if needed
        pass
    
    def record_compilation_end(self, job_id: str, success: bool, time_seconds: float, cache_hit: bool = False):
        """Record compilation completion"""
        with self._lock:
            self.compilation_metrics.record_compilation(success, time_seconds, cache_hit)
            logger.debug(f"Recorded compilation for {job_id}: success={success}, time={time_seconds}s, cache_hit={cache_hit}")
    
    # Profiling metrics
    def record_profiling_start(self, job_id: str):
        """Record profiling start (for tracking active profilings)"""
        # Could be used for tracking concurrent profilings if needed
        pass
    
    def record_profiling_end(self, job_id: str, success: bool, time_seconds: float, 
                            gpu_id: Optional[int], correctness: Optional[bool]):
        """Record profiling completion"""
        with self._lock:
            self.profiling_metrics.record_profiling(success, time_seconds, gpu_id, correctness)
            logger.debug(f"Recorded profiling for {job_id}: success={success}, time={time_seconds}s, gpu={gpu_id}, correct={correctness}")
    
    # GPU utilization metrics
    def record_gpu_acquisition(self, job_id: str, gpu_id: int, wait_time_seconds: float):
        """Record GPU acquisition"""
        with self._lock:
            self.gpu_utilization_metrics.record_gpu_acquisition(job_id, gpu_id, wait_time_seconds)
            logger.debug(f"Recorded GPU acquisition for {job_id}: gpu={gpu_id}, wait={wait_time_seconds}s")
    
    def record_gpu_release(self, job_id: str):
        """Record GPU release"""
        with self._lock:
            self.gpu_utilization_metrics.record_gpu_release(job_id)
            logger.debug(f"Recorded GPU release for {job_id}")
    
    # Throughput metrics
    def record_request_start(self, job_id: str):
        """Record request start"""
        with self._lock:
            self.throughput_metrics.record_request_start()
            logger.debug(f"Recorded request start for {job_id}")
    
    def record_request_completion(self, job_id: str, success: bool, end_to_end_time_seconds: float):
        """Record request completion"""
        with self._lock:
            self.throughput_metrics.record_request_completion(success, end_to_end_time_seconds)
            logger.debug(f"Recorded request completion for {job_id}: success={success}, time={end_to_end_time_seconds}s")
    
    # Get comprehensive stats
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get all collected metrics in a comprehensive report"""
        with self._lock:
            return {
                "compilation_metrics": self.compilation_metrics.get_stats(),
                "profiling_metrics": self.profiling_metrics.get_stats(),
                "gpu_utilization": self.gpu_utilization_metrics.get_stats(),
                "throughput": self.throughput_metrics.get_stats(),
                "timestamp": time.time()
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self.compilation_metrics = CompilationMetrics()
            self.profiling_metrics = ProfilingMetrics()
            self.gpu_utilization_metrics = GPUUtilizationMetrics()
            self.throughput_metrics = ThroughputMetrics()
            logger.info("All metrics reset")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get high-level summary statistics"""
        stats = self.get_comprehensive_stats()
        
        return {
            "server_uptime_minutes": stats["throughput"]["server_uptime_minutes"],
            "total_requests": stats["throughput"]["total_requests"],
            "requests_per_minute": stats["throughput"]["requests_per_minute"],
            "overall_success_rate": stats["throughput"]["success_rate"],
            "compilation_success_rate": stats["compilation_metrics"]["success_rate"],
            "profiling_success_rate": stats["profiling_metrics"]["success_rate"],
            "cache_hit_rate": stats["compilation_metrics"]["cache_hit_rate"],
            "average_compilation_time": stats["compilation_metrics"]["average_time_seconds"],
            "average_profiling_time": stats["profiling_metrics"]["average_time_seconds"],
            "average_gpu_wait_time": stats["gpu_utilization"]["average_wait_time_seconds"],
            "active_gpu_sessions": stats["gpu_utilization"]["active_sessions"]
        }


# Global metrics collector instance
_global_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def reset_global_metrics():
    """Reset the global metrics collector (useful for testing)"""
    global _global_metrics_collector
    if _global_metrics_collector is not None:
        _global_metrics_collector.reset_metrics()
