#!/usr/bin/env python3
"""
Unified Test Suite for Server Failure Handling
Tests all failure scenarios: HTTP 200 with failures, HTTP 400 client errors, 
subprocess crashes, and invalid requests
"""

import sys
import time
import json
import requests
import argparse
import logging
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Result Container
# ============================================================================

@dataclass
class TestResult:
    """Container for test results"""
    name: str
    category: str
    passed: bool
    duration: float
    details: str = ""


# ============================================================================
# Common Utilities
# ============================================================================

def check_server_health(server_url: str) -> bool:
    """Check if server is healthy"""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ Server is healthy")
            return True
        else:
            logger.error(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Cannot connect to server at {server_url}. Is it running?")
        return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


def make_request(server_url: str, request_data: Dict[str, Any], timeout: int = 35) -> Tuple[requests.Response, float]:
    """Make a request and return response with duration"""
    start_time = time.time()
    response = requests.post(
        f"{server_url}/compare",
        json=request_data,
        timeout=timeout
    )
    duration = time.time() - start_time
    return response, duration


# ============================================================================
# Kernel Factory Functions
# ============================================================================

class KernelFactory:
    """Factory for creating various kernel types"""
    
    @staticmethod
    def valid_kernel() -> Dict[str, Any]:
        """Create a valid kernel"""
        return {
            "source_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 2

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]
""",
            "kernel_type": "torch"
        }
    
    @staticmethod
    def syntax_error_kernel() -> Dict[str, Any]:
        """Create a kernel with syntax error"""
        return {
            "source_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * * 2  # Syntax error
""",
            "kernel_type": "torch"
        }
    
    @staticmethod
    def wrong_output_kernel() -> Dict[str, Any]:
        """Create a kernel with wrong output"""
        return {
            "source_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 3  # Wrong: should be * 2

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]
""",
            "kernel_type": "torch"
        }
    
    @staticmethod
    def missing_function_kernel() -> Dict[str, Any]:
        """Create a kernel missing required function"""
        return {
            "source_code": """
import torch

# Missing Model class or kernel function
def some_other_function(x):
    return x * 2
""",
            "kernel_type": "torch"
        }
    
    @staticmethod
    def value_error_kernel() -> Dict[str, Any]:
        """Create a kernel that raises ValueError"""
        return {
            "source_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise ValueError("Intentional ValueError")

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]
""",
            "kernel_type": "torch"
        }
    
    @staticmethod
    def segfault_kernel() -> Dict[str, Any]:
        """Create a kernel that will segfault"""
        return {
            "source_code": """
import torch
import ctypes

def kernel(x):
    ctypes.string_at(0)  # Causes segfault
    return x * 2
""",
            "kernel_type": "torch"
        }


# ============================================================================
# Request Factory Functions
# ============================================================================

class RequestFactory:
    """Factory for creating various request types"""
    
    @staticmethod
    def valid_request() -> Dict[str, Any]:
        """Create a valid request"""
        return {
            "ref_kernel": KernelFactory.valid_kernel(),
            "custom_kernel": KernelFactory.valid_kernel(),
            "num_trials": 10,
            "timeout": 30
        }
    
    @staticmethod
    def invalid_enum_request() -> Dict[str, Any]:
        """Create request with invalid enum value"""
        return {
            "ref_kernel": {
                "source_code": "def kernel(x): return x * 2",
                "kernel_type": "INVALID_TYPE"  # Invalid enum
            },
            "custom_kernel": KernelFactory.valid_kernel(),
            "num_trials": 10
        }
    
    @staticmethod
    def missing_field_request() -> Dict[str, Any]:
        """Create request with missing required field"""
        return {
            "ref_kernel": {
                "source_code": "def kernel(x): return x * 2",
                # Missing kernel_type
            },
            "custom_kernel": KernelFactory.valid_kernel()
        }
    
    @staticmethod
    def invalid_num_trials_request() -> Dict[str, Any]:
        """Create request with invalid num_trials"""
        return {
            "ref_kernel": KernelFactory.valid_kernel(),
            "custom_kernel": KernelFactory.valid_kernel(),
            "num_trials": 0  # Invalid: must be at least 1
        }
    
    @staticmethod
    def invalid_timeout_request() -> Dict[str, Any]:
        """Create request with invalid timeout"""
        return {
            "ref_kernel": KernelFactory.valid_kernel(),
            "custom_kernel": KernelFactory.valid_kernel(),
            "num_trials": 10,
            "timeout": 3  # Invalid: must be at least 5
        }
    
    @staticmethod
    def triton_no_io_request() -> Dict[str, Any]:
        """Create Triton request without IOContract"""
        return {
            "ref_kernel": KernelFactory.valid_kernel(),
            "custom_kernel": {
                "source_code": """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pass
""",
                "kernel_type": "triton"
                # Missing io field
            },
            "num_trials": 10
        }


# ============================================================================
# Test Validators
# ============================================================================

class TestValidator:
    """Validators for different response types"""
    
    @staticmethod
    def validate_200_success(response: requests.Response, duration: float) -> TestResult:
        """Validate successful 200 response"""
        if response.status_code != 200:
            return TestResult("", "", False, duration, f"Expected 200, got {response.status_code}")
        
        result = response.json()
        if result.get("status") != "success":
            return TestResult("", "", False, duration, f"Expected success status")
        
        kernel_result = result.get("kernel_exec_result", {})
        if not kernel_result.get("compiled") or not kernel_result.get("correctness"):
            return TestResult("", "", False, duration, "Expected compiled=true and correctness=true")
        
        return TestResult("", "", True, duration, "Valid request handled correctly")
    
    @staticmethod
    def validate_200_with_failure(response: requests.Response, duration: float) -> TestResult:
        """Validate 200 response with failure flags"""
        if response.status_code != 200:
            return TestResult("", "", False, duration, f"Expected 200, got {response.status_code}")
        
        result = response.json()
        if result.get("status") != "success":
            return TestResult("", "", False, duration, "Expected success status")
        
        kernel_result = result.get("kernel_exec_result", {})
        compiled = kernel_result.get("compiled", True)
        correctness = kernel_result.get("correctness", True)
        
        # Should have at least one failure
        if compiled and correctness:
            return TestResult("", "", False, duration, "Expected compilation or validation failure")
        
        details = f"Compiled: {compiled}, Correctness: {correctness}"
        if not compiled and kernel_result.get("compilation_error"):
            details += f", Error: {kernel_result['compilation_error'][:50]}..."
        
        return TestResult("", "", True, duration, details)
    
    @staticmethod
    def validate_400_client_error(response: requests.Response, duration: float) -> TestResult:
        """Validate 400 client error response"""
        if response.status_code != 400:
            return TestResult("", "", False, duration, f"Expected 400, got {response.status_code}")
        
        result = response.json()
        if result.get("status") != "client_error":
            return TestResult("", "", False, duration, "Expected client_error status")
        
        error_msg = result.get("error", "")
        validation_errors = result.get("validation_errors", [])
        
        details = f"Error: {error_msg}"
        if validation_errors:
            details += f", Fields: {[e['field'] for e in validation_errors]}"
        
        return TestResult("", "", True, duration, details)
    
    @staticmethod
    def validate_crash_handled(response: requests.Response, duration: float) -> TestResult:
        """Validate that crash was handled gracefully"""
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return TestResult("", "", True, duration, "Crash handled gracefully with 200")
        elif response.status_code == 500:
            # Quick 500 is acceptable for crashes
            if duration < 10:
                return TestResult("", "", True, duration, f"Quick failure in {duration:.2f}s")
            else:
                return TestResult("", "", False, duration, f"Took too long: {duration:.2f}s")
        
        # Timeout is bad
        if duration > 30:
            return TestResult("", "", False, duration, "Request timed out")
        
        return TestResult("", "", False, duration, f"Unexpected response: {response.status_code}")


# ============================================================================
# Test Runner
# ============================================================================

class TestRunner:
    """Main test runner"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.results: List[TestResult] = []
    
    def run_test(self, name: str, category: str, request_data: Dict[str, Any], 
                 validator: Callable) -> TestResult:
        """Run a single test"""
        logger.info(f"\n🔍 Testing {name}...")
        
        try:
            response, duration = make_request(self.server_url, request_data)
            result = validator(response, duration)
            result.name = name
            result.category = category
            
            if result.passed:
                logger.info(f"  ✅ PASSED in {duration:.2f}s")
                if result.details:
                    logger.info(f"     {result.details}")
            else:
                logger.error(f"  ❌ FAILED: {result.details}")
            
            return result
            
        except requests.Timeout:
            return TestResult(name, category, False, 35.0, "Request timed out")
        except Exception as e:
            return TestResult(name, category, False, 0.0, f"Exception: {e}")
    
    def run_test_suite(self, suite_name: str, tests: List[Tuple]) -> None:
        """Run a test suite"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        for test_name, request_factory, validator in tests:
            request_data = request_factory()
            result = self.run_test(test_name, suite_name, request_data, validator)
            self.results.append(result)
    
    def print_summary(self) -> bool:
        """Print test summary and return success status"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Print by category
        total_passed = 0
        total_tests = len(self.results)
        
        for category, results in categories.items():
            print(f"\n{category}:")
            for result in results:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {result.name}")
            
            passed = sum(1 for r in results if r.passed)
            total_passed += passed
            print(f"  Subtotal: {passed}/{len(results)} passed")
        
        print(f"\n{'='*60}")
        print(f"Total: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("\n🎉 All tests passed! Server failure handling is working correctly.")
            return True
        else:
            print("\n❌ Some tests failed. Check the implementation.")
            return False


# ============================================================================
# Test Suites Definition
# ============================================================================

def get_test_suites():
    """Define all test suites"""
    return {
        "HTTP 200 Success Tests": [
            ("Valid Request", RequestFactory.valid_request, TestValidator.validate_200_success),
        ],
        "HTTP 200 Failure Tests": [
            ("Compilation Failure", 
             lambda: {"ref_kernel": KernelFactory.valid_kernel(), 
                     "custom_kernel": KernelFactory.syntax_error_kernel(),
                     "num_trials": 10, "timeout": 30},
             TestValidator.validate_200_with_failure),
            ("Validation Failure",
             lambda: {"ref_kernel": KernelFactory.valid_kernel(),
                     "custom_kernel": KernelFactory.wrong_output_kernel(),
                     "num_trials": 10, "timeout": 30},
             TestValidator.validate_200_with_failure),
            ("Missing Function",
             lambda: {"ref_kernel": KernelFactory.valid_kernel(),
                     "custom_kernel": KernelFactory.missing_function_kernel(),
                     "num_trials": 10, "timeout": 30},
             TestValidator.validate_200_with_failure),
            ("ValueError in Kernel",
             lambda: {"ref_kernel": KernelFactory.valid_kernel(),
                     "custom_kernel": KernelFactory.value_error_kernel(),
                     "num_trials": 10, "timeout": 30},
             TestValidator.validate_200_with_failure),
        ],
        "HTTP 400 Client Error Tests": [
            ("Invalid Enum Value", RequestFactory.invalid_enum_request, TestValidator.validate_400_client_error),
            ("Missing Required Field", RequestFactory.missing_field_request, TestValidator.validate_400_client_error),
            ("Invalid num_trials", RequestFactory.invalid_num_trials_request, TestValidator.validate_400_client_error),
            ("Invalid timeout", RequestFactory.invalid_timeout_request, TestValidator.validate_400_client_error),
            ("Triton without IOContract", RequestFactory.triton_no_io_request, TestValidator.validate_400_client_error),
        ],
        "Crash Handling Tests": [
            ("Segmentation Fault",
             lambda: {"ref_kernel": KernelFactory.valid_kernel(),
                     "custom_kernel": KernelFactory.segfault_kernel(),
                     "num_trials": 10, "timeout": 30},
             TestValidator.validate_crash_handled),
        ],
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified test suite for server failure handling"
    )
    parser.add_argument(
        "server_url",
        help="Server URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--suite",
        choices=["all", "http200", "http400", "crashes"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "="*60)
    print("🔍 Unified Server Failure Handling Test Suite")
    print("="*60)
    print(f"Server: {args.server_url}")
    print(f"Suite: {args.suite}")
    
    # Check server health
    if not check_server_health(args.server_url):
        sys.exit(1)
    
    # Initialize test runner
    runner = TestRunner(args.server_url)
    
    # Get test suites
    all_suites = get_test_suites()
    
    # Select suites to run
    if args.suite == "all":
        suites_to_run = all_suites
    elif args.suite == "http200":
        suites_to_run = {k: v for k, v in all_suites.items() if "200" in k}
    elif args.suite == "http400":
        suites_to_run = {k: v for k, v in all_suites.items() if "400" in k}
    elif args.suite == "crashes":
        suites_to_run = {k: v for k, v in all_suites.items() if "Crash" in k}
    
    # Run selected suites
    for suite_name, tests in suites_to_run.items():
        runner.run_test_suite(suite_name, tests)
    
    # Print summary and exit
    success = runner.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()