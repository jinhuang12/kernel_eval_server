"""
Request and response factories for testing
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from shared.models import KernelType
from .test_data_loader import get_loader

@dataclass
class RequestFactory:
    """Factory for creating test requests"""
    
    @staticmethod
    def create_evaluate_request(
        kernel: Optional[Dict[str, Any]] = None,
        num_trials: int = 10,
        timeout: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """Create an evaluate request"""
        request = {
            "num_trials": num_trials,
            "timeout": timeout
        }
        
        if kernel:
            # Convert kernel format to match API
            api_kernel = {
                "kernel_type": kernel.get("kernel_type", "torch"),
                "source_code": kernel.get("source_code", kernel.get("kernel_code", "")),
            }
            if kernel.get("io"):
                api_kernel["io"] = kernel["io"]
            elif kernel.get("io_contract"):
                api_kernel["io"] = kernel["io_contract"]
            if kernel.get("metadata"):
                api_kernel["metadata"] = kernel["metadata"]
            request["kernel"] = api_kernel
            
        # Add any additional fields
        request.update(kwargs)
        
        return request
    
    @staticmethod
    def create_compare_request(
        ref_kernel: Optional[Dict[str, Any]] = None,
        custom_kernel: Optional[Dict[str, Any]] = None,
        num_trials: int = 10,
        timeout: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a compare request"""
        request = {
            "num_trials": num_trials,
            "timeout": timeout
        }
        
        if ref_kernel:
            # Convert kernel format to match API
            api_kernel = {
                "kernel_type": ref_kernel.get("kernel_type", "torch"),
                "source_code": ref_kernel.get("source_code", ref_kernel.get("kernel_code", "")),
            }
            if ref_kernel.get("io"):
                api_kernel["io"] = ref_kernel["io"]
            elif ref_kernel.get("io_contract"):
                api_kernel["io"] = ref_kernel["io_contract"]
            if ref_kernel.get("metadata"):
                api_kernel["metadata"] = ref_kernel["metadata"]
            request["ref_kernel"] = api_kernel
            
        if custom_kernel:
            # Convert kernel format to match API
            api_kernel = {
                "kernel_type": custom_kernel.get("kernel_type", "torch_cuda"),
                "source_code": custom_kernel.get("source_code", custom_kernel.get("kernel_code", "")),
            }
            if custom_kernel.get("io"):
                api_kernel["io"] = custom_kernel["io"]
            elif custom_kernel.get("io_contract"):
                api_kernel["io"] = custom_kernel["io_contract"]
            if custom_kernel.get("metadata"):
                api_kernel["metadata"] = custom_kernel["metadata"]
            request["custom_kernel"] = api_kernel
            
        request.update(kwargs)
        
        return request
    
    @staticmethod
    def create_kernel_request(
        kernel_type: str,
        kernel_name: str,
        kernel_code: str,
        io_contract: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a kernel definition"""
        kernel = {
            "kernel_type": kernel_type,
            "kernel_name": kernel_name,
            "kernel_code": kernel_code
        }
        
        if io_contract:
            kernel["io_contract"] = io_contract
            
        kernel.update(kwargs)
        
        return kernel
    
    @staticmethod
    def create_from_test_case(test_case_name: str, endpoint: str = "compare") -> Optional[Dict[str, Any]]:
        """Create a request from a test case file"""
        loader = get_loader()
        test_case = loader.get_test_case(test_case_name)
        
        if not test_case:
            return None
        
        if endpoint == "evaluate":
            # For evaluate, extract custom kernel
            kernel = test_case.get("custom_kernel") or test_case.get("ref_kernel")
            if not kernel:
                return None
            return RequestFactory.create_evaluate_request(
                kernel=kernel,
                num_trials=test_case.get("num_trials", 10),
                timeout=test_case.get("timeout", 30)
            )
        else:
            # For compare, use both kernels
            return RequestFactory.create_compare_request(
                ref_kernel=test_case.get("ref_kernel"),
                custom_kernel=test_case.get("custom_kernel"),
                num_trials=test_case.get("num_trials", 10),
                timeout=test_case.get("timeout", 30)
            )
    
    @staticmethod
    def get_all_test_requests(kernel_type: str = None, endpoint: str = "compare") -> List[Tuple[str, Dict[str, Any]]]:
        """Get all test requests from test data files"""
        loader = get_loader()
        test_cases = loader.get_all_test_cases(kernel_type)
        
        requests = []
        for name, test_case in test_cases:
            if endpoint == "evaluate":
                kernel = test_case.get("custom_kernel") or test_case.get("ref_kernel")
                if kernel:
                    request = RequestFactory.create_evaluate_request(
                        kernel=kernel,
                        num_trials=test_case.get("num_trials", 10),
                        timeout=test_case.get("timeout", 30)
                    )
                    requests.append((name, request))
            else:
                request = RequestFactory.create_compare_request(
                    ref_kernel=test_case.get("ref_kernel"),
                    custom_kernel=test_case.get("custom_kernel"),
                    num_trials=test_case.get("num_trials", 10),
                    timeout=test_case.get("timeout", 30)
                )
                requests.append((name, request))
        
        return requests
    
    @staticmethod
    def create_invalid_request() -> List[Dict[str, Any]]:
        """Create various invalid requests for testing"""
        return [
            # Missing required fields
            {"index": "test"},
            
            # Invalid enum value
            {
                "index": "test",
                "ref_kernel": {
                    "kernel_type": "invalid_type",
                    "kernel_name": "test",
                    "kernel_code": "pass"
                }
            },
            
            # Invalid num_trials
            {
                "index": "test",
                "num_trials": 0,
                "ref_kernel": {
                    "kernel_type": "torch",
                    "kernel_name": "test",
                    "kernel_code": "pass"
                }
            },
            
            # Invalid timeout
            {
                "index": "test",
                "timeout": 2,
                "ref_kernel": {
                    "kernel_type": "torch",
                    "kernel_name": "test",
                    "kernel_code": "pass"
                }
            },
            
            # Triton without IOContract
            {
                "index": "test",
                "ref_kernel": {
                    "kernel_type": "triton",
                    "kernel_name": "test",
                    "kernel_code": "pass"
                }
            }
        ]


@dataclass
class ResponseValidator:
    """Validator for API responses"""
    
    @staticmethod
    def validate_success_response(response: Dict[str, Any]) -> bool:
        """Validate a successful response"""
        required_fields = ["status", "job_id", "kernel_exec_result"]
        
        # Check required fields
        for field in required_fields:
            if field not in response:
                return False
        
        # Check status
        if response["status"] != "success":
            return False
        
        # Check kernel_exec_result structure
        ker = response["kernel_exec_result"]
        if "compiled" not in ker or "correctness" not in ker:
            return False
        
        return True
    
    @staticmethod
    def validate_error_response(response: Dict[str, Any], expected_status: int) -> bool:
        """Validate an error response"""
        if expected_status == 400:
            # Client error should have validation_errors or error message
            return "error" in response or "validation_errors" in response
        elif expected_status == 500:
            # Server error should have error message
            return "error" in response
        
        return False
    
    @staticmethod
    def validate_evaluate_response(response: Dict[str, Any]) -> bool:
        """Validate evaluate endpoint response"""
        if not ResponseValidator.validate_success_response(response):
            return False
        
        ker = response["kernel_exec_result"]
        
        # Check for runtime stats
        if "runtime_stats" in ker and ker["runtime_stats"]:
            stats = ker["runtime_stats"]
            if "mean" not in stats or "median" not in stats:
                return False
        
        return True
    
    @staticmethod
    def validate_compare_response(response: Dict[str, Any]) -> bool:
        """Validate compare endpoint response"""
        return ResponseValidator.validate_success_response(response)
    
    @staticmethod
    def extract_performance(response: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from response"""
        perf = {}
        
        # Extract from kernel_exec_result
        if "kernel_exec_result" in response:
            ker = response["kernel_exec_result"]
            if "runtime" in ker:
                perf["runtime"] = ker["runtime"]
            
            if "runtime_stats" in ker and ker["runtime_stats"]:
                stats = ker["runtime_stats"]
                perf["mean"] = stats.get("mean", 0)
                perf["median"] = stats.get("median", 0)
                perf["std"] = stats.get("std", 0)
                perf["min"] = stats.get("min", 0)
                perf["max"] = stats.get("max", 0)
        
        # For compare endpoint, extract ref_runtime
        if "ref_runtime" in response:
            ref = response["ref_runtime"]
            perf["ref_mean"] = ref.get("mean", 0)
            perf["ref_median"] = ref.get("median", 0)
            
            # Calculate speedup if both are available
            if perf.get("ref_mean", 0) > 0 and perf.get("mean", 0) > 0:
                perf["speedup"] = perf["ref_mean"] / perf["mean"]
        
        return perf
    
    @staticmethod
    def validate_compilation_success(response: Dict[str, Any]) -> bool:
        """Check if kernel compiled successfully"""
        if "kernel_exec_result" not in response:
            return False
        return response["kernel_exec_result"].get("compiled", False)
    
    @staticmethod
    def validate_correctness(response: Dict[str, Any]) -> bool:
        """Check if kernel output is correct"""
        if "kernel_exec_result" not in response:
            return False
        return response["kernel_exec_result"].get("correctness", False)
    
    @staticmethod
    def validate_performance_metrics(response: Dict[str, Any]) -> bool:
        """Check if performance metrics are present"""
        if "kernel_exec_result" not in response:
            return False
        ker = response["kernel_exec_result"]
        
        # Must have runtime
        if "runtime" not in ker or ker["runtime"] <= 0:
            return False
        
        # If runtime_stats present, validate structure
        if "runtime_stats" in ker and ker["runtime_stats"]:
            stats = ker["runtime_stats"]
            required = ["mean", "median", "min", "max", "std"]
            return all(field in stats for field in required)
        
        return True
    
    @staticmethod
    def validate_speedup(response: Dict[str, Any]) -> bool:
        """Check if speedup calculation is present (for compare endpoint)"""
        perf = ResponseValidator.extract_performance(response)
        return "speedup" in perf and perf["speedup"] > 0
    
    @staticmethod
    def validate_device_metrics(response: Dict[str, Any]) -> bool:
        """Check if device metrics are present and properly structured"""
        if "kernel_exec_result" not in response:
            return False
        
        ker = response["kernel_exec_result"]
        if "metadata" not in ker:
            return False
        
        metadata = ker["metadata"]
        if "device_metrics" not in metadata or not metadata["device_metrics"]:
            return False
        
        metrics = metadata["device_metrics"]
        
        # Check for at least one category of metrics
        categories = ["speed_of_light", "detailed_metrics", "memory_metrics", 
                     "compute_metrics", "pipeline_metrics", "occupancy_metrics"]
        
        return any(cat in metrics and metrics[cat] for cat in categories)


@dataclass
class TestResult:
    """Container for test results"""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        msg = f"{status} - {self.name} ({self.duration:.2f}s)"
        if self.error:
            msg += f"\n  Error: {self.error}"
        return msg


@dataclass  
class TestSuite:
    """Collection of related tests"""
    name: str
    tests: List[TestResult] = field(default_factory=list)
    
    def add_test(self, result: TestResult):
        """Add a test result"""
        self.tests.append(result)
    
    def passed_count(self) -> int:
        """Count of passed tests"""
        return sum(1 for t in self.tests if t.passed)
    
    def failed_count(self) -> int:
        """Count of failed tests"""
        return sum(1 for t in self.tests if not t.passed)
    
    def total_count(self) -> int:
        """Total test count"""
        return len(self.tests)
    
    def success_rate(self) -> float:
        """Success rate percentage"""
        if not self.tests:
            return 0.0
        return (self.passed_count() / self.total_count()) * 100
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"{self.name}: {self.passed_count()}/{self.total_count()} passed "
            f"({self.success_rate():.1f}%)"
        )