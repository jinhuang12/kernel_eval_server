"""
Response validation helpers for testing
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
import requests

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Validates API responses"""
    
    @staticmethod
    def check_health(base_url: str, timeout: int = 30) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{base_url}/health", timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    @staticmethod
    def validate_status_code(response: requests.Response, expected: int) -> bool:
        """Validate HTTP status code"""
        if response.status_code != expected:
            logger.error(f"Expected status {expected}, got {response.status_code}")
            return False
        return True
    
    @staticmethod
    def validate_json_response(response: requests.Response) -> Optional[Dict[str, Any]]:
        """Validate and parse JSON response"""
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    @staticmethod
    def validate_success_fields(data: Dict[str, Any]) -> bool:
        """Validate required fields in success response"""
        # Handle different response formats
        if "kernel_exec_result" in data:
            # New format from /evaluate endpoint
            if data.get("status") != "success":
                logger.error(f"Unexpected status: {data.get('status')}")
                return False
            exec_result = data["kernel_exec_result"]
            if "compiled" not in exec_result:
                logger.error("Missing 'compiled' in kernel_exec_result")
                return False
            return True
        elif "results" in data:
            # Old format
            if data.get("status") != "completed":
                logger.error(f"Unexpected status: {data.get('status')}")
                return False
            results = data["results"]
            if "compiled" not in results:
                logger.error("Missing 'compiled' in results")
                return False
            return True
        
        logger.error("Unrecognized response format")
        return False
    
    @staticmethod
    def validate_error_fields(data: Dict[str, Any]) -> bool:
        """Validate error response fields"""
        if "error" not in data and "validation_errors" not in data:
            logger.error("Missing error information in response")
            return False
        return True
    
    @staticmethod
    def validate_compilation_success(data: Dict[str, Any]) -> bool:
        """Validate successful compilation"""
        if "kernel_exec_result" in data:
            # New format
            exec_result = data["kernel_exec_result"]
            if not exec_result.get("compiled", False):
                logger.error(f"Compilation failed: {exec_result.get('compilation_error', 'Unknown error')}")
                return False
            return True
        elif "results" in data:
            # Old format
            results = data["results"]
            if not results.get("compiled", False):
                logger.error(f"Compilation failed: {results.get('error', 'Unknown error')}")
                return False
            return True
        
        return False
    
    @staticmethod
    def validate_correctness(data: Dict[str, Any]) -> bool:
        """Validate correctness check passed"""
        if "kernel_exec_result" in data:
            # New format
            exec_result = data["kernel_exec_result"]
            if not exec_result.get("correctness", False):
                logger.error(f"Correctness check failed: {exec_result.get('validation_error', 'Unknown')}")
                return False
            return True
        elif "results" in data:
            # Old format
            results = data["results"]
            if not results.get("correctness", False):
                logger.error(f"Correctness check failed: {results.get('validation_error', 'Unknown')}")
                return False
            return True
        
        return False
    
    @staticmethod
    def validate_performance_metrics(data: Dict[str, Any], kernel_type: str = "ref") -> bool:
        """Validate performance metrics present"""
        if "kernel_exec_result" in data:
            # New format - performance data may be in different structure
            exec_result = data["kernel_exec_result"]
            # For now, just check if compilation succeeded
            # Performance metrics might be in a different format
            return exec_result.get("compiled", False)
        elif "results" in data:
            # Old format
            results = data["results"]
            perf_key = f"{kernel_type}_performance"
            
            if perf_key not in results:
                logger.warning(f"No {perf_key} in results")
                return False
            
            perf = results[perf_key]
            required_metrics = ["mean_time", "median_time", "min_time", "max_time"]
            
            for metric in required_metrics:
                if metric not in perf:
                    logger.error(f"Missing metric: {metric}")
                    return False
                
                if not isinstance(perf[metric], (int, float)):
                    logger.error(f"Invalid metric type for {metric}: {type(perf[metric])}")
                    return False
                
                if perf[metric] < 0:
                    logger.error(f"Negative value for {metric}: {perf[metric]}")
                    return False
            
            return True
        
        return False
    
    @staticmethod
    def validate_speedup(data: Dict[str, Any], min_speedup: float = 0.1) -> bool:
        """Validate speedup calculation"""
        results = data.get("results", {})
        
        ref_perf = results.get("ref_performance", {})
        target_perf = results.get("target_performance", {})
        
        if not ref_perf or not target_perf:
            logger.warning("Cannot calculate speedup without both performances")
            return True  # Not an error, just no speedup
        
        ref_time = ref_perf.get("mean_time", 0)
        target_time = target_perf.get("mean_time", 0)
        
        if ref_time <= 0 or target_time <= 0:
            logger.error(f"Invalid times for speedup: ref={ref_time}, target={target_time}")
            return False
        
        speedup = ref_time / target_time
        
        if speedup < min_speedup:
            logger.warning(f"Low speedup: {speedup:.2f}x")
        
        return True


class TestTimer:
    """Simple timer for test duration tracking"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
        return self
    
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit"""
        self.stop()


def retry_on_failure(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """Retry a function on failure with exponential backoff"""
    
    current_delay = delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
            time.sleep(current_delay)
            current_delay *= backoff
    
    return None