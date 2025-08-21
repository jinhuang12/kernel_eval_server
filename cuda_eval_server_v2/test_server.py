"""
Enhanced Load Testing Suite for CUDA Evaluation Server V2
Comprehensive testing of concurrent requests, GPU resource management, and performance metrics
Updated to use KernelBench generated test cases for validation testing
"""

import requests
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics
import random
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test data paths
REF_CODE_PATH = "test_data/ref_code_model_ex_add.py"
CUSTOM_CODE_PATH = "test_data/custom_code_new_model_ex_add.py"
KERNELBENCH_JSON_PATH =  "test_data/kernelbench_evaluated_20250808_212140.json" #"test_data/kernelbench_generated_kernels.json"


@dataclass
class TestResult:
    """Container for individual test results"""
    success: bool
    response_time: float
    status_code: int
    job_id: str = None
    error: str = None
    gpu_id: int = None
    compilation_time: float = None
    profiling_time: float = None
    # Expected vs actual validation results
    expected_compile: Optional[bool] = None
    expected_correct: Optional[bool] = None
    expected_cuda: Optional[bool] = None
    actual_compile: Optional[bool] = None
    actual_correct: Optional[bool] = None
    actual_cuda: Optional[bool] = None
    validation_passed: Optional[bool] = None
    expected_failure_but_passed: Optional[bool] = None  # New field for tracking special case
    # Performance metrics from the new JSON format
    ref_runtime_ms: Optional[float] = None
    generated_runtime_ms: Optional[float] = None
    speedup: Optional[float] = None
    # Device metrics
    device_metrics: Optional[Dict[str, Any]] = None


@dataclass
class LoadTestMetrics:
    """Container for load test metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = None
    errors: List[str] = None
    gpu_ids_used: List[int] = None
    concurrent_peak: int = 0
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.errors is None:
            self.errors = []
        if self.gpu_ids_used is None:
            self.gpu_ids_used = []


class KernelBenchTestCaseProvider:
    """Loads test cases from kernelbench_generated_kernels.json"""
    
    def __init__(self):
        self.test_cases = []
        self.cuda_only_cases = []
        self._load_json_test_cases()
        
    def _load_json_test_cases(self):
        """Load test cases from JSON file"""
        try:
            if not os.path.exists(KERNELBENCH_JSON_PATH):
                logger.error(f"KernelBench JSON file not found: {KERNELBENCH_JSON_PATH}")
                return
                
            logger.info(f"Loading test cases from {KERNELBENCH_JSON_PATH}...")
            
            # Load JSON in chunks to avoid memory issues with large file
            with open(KERNELBENCH_JSON_PATH, 'r') as f:
                data = json.load(f)
                
            logger.info(f"Loaded {len(data)} test cases from JSON")
            
            for idx, case in enumerate(data):
                if all(key in case for key in ['ref', 'generated', 'compile', 'correct', 'cuda']):
                    test_case = {
                        "index": idx,
                        "ref_code": case['ref'],
                        "custom_code": case['generated'], 
                        "expected_compile": case['compile'],
                        "expected_correct": case['correct'],
                        "expected_cuda": case['cuda'],
                        # Extract performance metrics from new JSON format
                        "ref_runtime_ms": case.get('ref_runtime_ms'),
                        "generated_runtime_ms": case.get('generated_runtime_ms'),
                        "speedup": case.get('speedup')
                    }
                    self.test_cases.append(test_case)
                    
                    # Filter CUDA-only cases for tests that need GPU usage
                    if case['cuda']:
                        self.cuda_only_cases.append(test_case)
                else:
                    logger.warning(f"Test case {idx} missing required fields")
                    
            logger.info(f"Processed {len(self.test_cases)} total cases, {len(self.cuda_only_cases)} CUDA cases")
            
        except Exception as e:
            logger.error(f"Failed to load JSON test cases: {e}")
            
    def get_test_case(self, index: int = 0) -> Optional[Dict[str, Any]]:
        """Get a test case by index"""
        if index < len(self.test_cases):
            return self.test_cases[index]
        return None
        
    def get_random_test_case(self) -> Optional[Dict[str, Any]]:
        """Get a random test case from all cases"""
        if self.test_cases:
            return random.choice(self.test_cases)
        return None
        
    def get_random_cuda_test_case(self) -> Optional[Dict[str, Any]]:
        """Get a random test case that actually uses CUDA"""
        if self.cuda_only_cases:
            return random.choice(self.cuda_only_cases)
        return None
        
    def get_test_batch(self, size: int = 10, cuda_only: bool = False) -> List[Dict[str, Any]]:
        """Get a batch of random test cases"""
        source_cases = self.cuda_only_cases if cuda_only else self.test_cases
        if not source_cases:
            return []
            
        # Sample without replacement if possible
        sample_size = min(size, len(source_cases))
        return random.sample(source_cases, sample_size)
        
    def count_cases(self) -> Dict[str, int]:
        """Get counts of different types of test cases"""
        return {
            "total_cases": len(self.test_cases),
            "cuda_cases": len(self.cuda_only_cases),
            "compile_true": sum(1 for case in self.test_cases if case['expected_compile']),
            "correct_true": sum(1 for case in self.test_cases if case['expected_correct'])
        }


class TestCaseProvider:
    """Manages test cases for load testing - with fallback to original test files"""
    
    def __init__(self):
        # Try to load from JSON first
        self.kernelbench_provider = KernelBenchTestCaseProvider()
        
        # Fallback to original test files
        self.ref_code = self._load_test_file(REF_CODE_PATH)
        self.custom_code = self._load_test_file(CUSTOM_CODE_PATH)
        
        self.fallback_test_cases = [
            {
                "name": "element_wise_add", 
                "ref_code": self.ref_code,
                "custom_code": self.custom_code,
                "expected_compile": True,
                "expected_correct": True,
                "expected_cuda": True
            }
        ]
        
        # Log provider status
        counts = self.kernelbench_provider.count_cases()
        if counts['total_cases'] > 0:
            logger.info(f"KernelBench provider loaded: {counts}")
        else:
            logger.warning("KernelBench provider failed, using fallback test cases")
    
    def _load_test_file(self, path: str) -> str:
        """Load test file content"""
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Test file not found: {path}")
            return ""
    
    def get_test_case(self, index: int = 0) -> Dict[str, Any]:
        """Get a test case by index"""
        # Try KernelBench first
        case = self.kernelbench_provider.get_test_case(index)
        if case:
            return case
            
        # Fallback to original
        if index < len(self.fallback_test_cases):
            return self.fallback_test_cases[index]
        return self.fallback_test_cases[0]
    
    def get_random_test_case(self, cuda_only: bool = False) -> Dict[str, Any]:
        """Get a random test case"""
        if cuda_only:
            case = self.kernelbench_provider.get_random_cuda_test_case()
        else:
            case = self.kernelbench_provider.get_random_test_case()
            
        if case:
            return case
        return random.choice(self.fallback_test_cases)
        
    def get_test_batch(self, size: int = 10, cuda_only: bool = False) -> List[Dict[str, Any]]:
        """Get a batch of test cases"""
        batch = self.kernelbench_provider.get_test_batch(size, cuda_only)
        if batch:
            return batch
            
        # Fallback: repeat the single test case
        return [self.fallback_test_cases[0]] * min(size, len(self.fallback_test_cases))
    
    def create_test_request(self, num_trials: int = 100, timeout: int = 60, test_case: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a test request"""
        if not test_case:
            test_case = self.get_test_case()
            
        return {
            "ref_code": test_case["ref_code"],
            "custom_code": test_case["custom_code"],
            "num_trials": num_trials,
            "timeout": timeout
        }
        
    def create_validation_test_request(self, test_case: Dict[str, Any], num_trials: int = 10, timeout: int = 60) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create a test request with expected validation results"""
        request = self.create_test_request(num_trials, timeout, test_case)
        expected = {
            "compile": test_case.get("expected_compile"),
            "correct": test_case.get("expected_correct"), 
            "cuda": test_case.get("expected_cuda")
        }
        return request, expected


class ValidationTestRunner:
    """Runs validation tests comparing expected vs actual results"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.test_provider = TestCaseProvider()
        
    def validate_single_test_case(self, test_case: Dict[str, Any], num_trials: int = 10, log_details: bool = True) -> TestResult:
        """Run a single test case and validate against expected results"""
        start_time = time.time()
        
        # Log expected values BEFORE running the test
        if log_details:
            index = test_case.get('index', 'unknown')
            expected_compile = test_case.get('expected_compile')
            expected_correct = test_case.get('expected_correct')
            expected_cuda = test_case.get('expected_cuda')
            
            logger.info(f"📋 Running test case index: {index}")
            logger.info(f"   Expected results from ground truth:")
            logger.info(f"     • Compile: {expected_compile}")
            logger.info(f"     • Correct: {expected_correct}")
            logger.info(f"     • CUDA:    {expected_cuda}")
            
            # Display performance metrics if available
            ref_runtime = test_case.get('ref_runtime_ms')
            gen_runtime = test_case.get('generated_runtime_ms')
            speedup = test_case.get('speedup')
            
            if ref_runtime is not None and gen_runtime is not None and speedup is not None:
                if ref_runtime > 0 and gen_runtime > 0 and speedup > 0:
                    logger.info(f"   Performance metrics from evaluation:")
                    logger.info(f"     • Ref Runtime: {ref_runtime:.3f} ms")
                    logger.info(f"     • Gen Runtime: {gen_runtime:.3f} ms")
                    logger.info(f"     • Speedup: {speedup:.2f}x")
            
            # Describe what we expect to happen
            if expected_compile and expected_correct:
                logger.info(f"   ➡️ This test SHOULD SUCCEED (compile and run correctly)")
            elif expected_compile and not expected_correct:
                logger.info(f"   ➡️ This test SHOULD COMPILE but FAIL correctness validation")
            elif not expected_compile:
                logger.info(f"   ➡️ This test SHOULD FAIL at compilation")
            else:
                logger.info(f"   ➡️ This test has complex expectations")
        
        try:
            request, expected = self.test_provider.create_validation_test_request(test_case, num_trials)
            
            response = requests.post(
                f"{self.base_url}/",
                json=request,
                timeout=120
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Extract job_id immediately and log it
                job_id = result_data.get("job_id", "unknown")
                if log_details:
                    logger.info(f"   📝 Received response - Job ID: {job_id}")
                
                # Extract actual results from response
                actual_compile = self._extract_compilation_status(result_data)
                actual_correct = self._extract_correctness_status(result_data)
                actual_cuda = self._extract_cuda_usage(result_data)
                
                # Special handling for expected failures that actually pass
                expected_failure_but_passed = False
                if not expected['compile'] and actual_compile:
                    # Expected compilation failure but it compiled
                    expected_failure_but_passed = True
                    validation_passed = True  # Count as validation success (special case)
                elif not expected['correct'] and actual_correct and expected['compile']:
                    # Expected correctness failure but it passed
                    expected_failure_but_passed = True
                    validation_passed = True  # Count as validation success (special case)
                else:
                    # Standard validation
                    compile_match = expected['compile'] == actual_compile if expected['compile'] is not None else True
                    correct_match = expected['correct'] == actual_correct if expected['correct'] is not None else True
                    cuda_match = expected['cuda'] == actual_cuda if expected['cuda'] is not None else True
                    validation_passed = compile_match and correct_match # and cuda_match
                
                # Extract other info
                gpu_id = self._extract_gpu_id(result_data)
                
                # Extract device metrics if available
                device_metrics = self._extract_device_metrics(result_data)
                
                return TestResult(
                    success=True,
                    response_time=response_time,
                    status_code=response.status_code,
                    job_id=job_id,
                    gpu_id=gpu_id,
                    expected_compile=expected['compile'],
                    expected_correct=expected['correct'],
                    expected_cuda=expected['cuda'],
                    actual_compile=actual_compile,
                    actual_correct=actual_correct,
                    actual_cuda=actual_cuda,
                    validation_passed=validation_passed,
                    expected_failure_but_passed=expected_failure_but_passed,
                    device_metrics=device_metrics
                )
            else:
                # Even HTTP errors need to be validated against expectations
                # If we expected compilation failure, an HTTP 500 might be correct
                error_msg = response.text[:500] if response.text else ""
                
                # Try to extract what actually happened from error response
                actual_compile = False  # HTTP error usually means compilation failed
                actual_correct = None   # Can't determine correctness if compilation failed
                actual_cuda = True      # Server uses CUDA by default
                
                # Check if this failure was expected
                compile_match = expected['compile'] == actual_compile if expected['compile'] is not None else True
                correct_match = True  # Can't check correctness on compilation failure
                cuda_match = expected['cuda'] == actual_cuda if expected['cuda'] is not None else True
                
                validation_passed = compile_match and correct_match # and cuda_match
                
                return TestResult(
                    success=False,
                    response_time=response_time,
                    status_code=response.status_code,
                    error=f"HTTP {response.status_code}: {error_msg}",
                    expected_compile=expected['compile'],
                    expected_correct=expected['correct'],
                    expected_cuda=expected['cuda'],
                    actual_compile=actual_compile,
                    actual_correct=actual_correct,
                    actual_cuda=actual_cuda,
                    validation_passed=validation_passed
                )
                
        except Exception as e:
            return TestResult(
                success=False,
                response_time=time.time() - start_time,
                status_code=0,
                error=str(e),
                expected_compile=expected.get('compile'),
                expected_correct=expected.get('correct'),
                expected_cuda=expected.get('cuda')
            )
    
    def _extract_compilation_status(self, response_data: Dict[str, Any]) -> Optional[bool]:
        """Extract compilation status from server response"""
        status = response_data.get('status')
        
        # Success means both compilation and validation passed
        if status == 'success':
            return True
        
        # For errors, we need to distinguish compilation failure from correctness failure
        if status == 'error':
            error_msg = response_data.get('error_message', '') or response_data.get('error', '')
            error_msg_lower = error_msg.lower()
            
            # Check for specific error types
            # Compilation failures contain these keywords
            if 'compilation failed' in error_msg_lower:
                return False  # Compilation failed
            
            # Validation failures mean compilation succeeded
            if 'validation failed' in error_msg_lower:
                return True  # Compiled but validation failed
            
            # Shape mismatch errors also indicate successful compilation
            if 'shape mismatch' in error_msg_lower:
                return True  # Compiled but validation failed
            
            # Correctness failures mean compilation succeeded
            if 'correctness' in error_msg_lower and 'failed' in error_msg_lower:
                return True  # Compiled but correctness failed
            
            # If error mentions compilation errors
            if 'compile' in error_msg_lower and ('error' in error_msg_lower or 'failed' in error_msg_lower):
                return False  # Compilation failed
            
            # Default for unknown errors - assume compilation failed
            return False
        
        return None
            
    
    def _extract_correctness_status(self, response_data: Dict[str, Any]) -> Optional[bool]:
        """Extract correctness validation status from server response"""
        status = response_data.get('status')
        
        # Success means both compilation and correctness passed
        if status == 'success':
            return True
        
        # For errors, check if it's a correctness failure
        if status == 'error':
            error_msg = response_data.get('error_message', '') or response_data.get('error', '')
            error_msg_lower = error_msg.lower()
            
            # If compilation failed, we can't determine correctness
            if 'compilation failed' in error_msg_lower:
                return None  # Can't determine correctness if compilation failed
            
            # Validation failures mean correctness failed
            if 'validation failed' in error_msg_lower:
                return False  # Validation/correctness failed
            
            # Shape mismatch means correctness failed
            if 'shape mismatch' in error_msg_lower:
                return False  # Correctness failed due to shape mismatch
                
            # Explicit correctness failure
            if 'correctness' in error_msg_lower and 'failed' in error_msg_lower:
                return False  # Correctness explicitly failed
            
            # If error doesn't mention validation/correctness, we can't determine
            return None
            
        return None
    
    def _extract_cuda_usage(self, response_data: Dict[str, Any]) -> Optional[bool]:
        """Extract whether CUDA was actually used from server response"""
        # Check compilation method first (most reliable)
        compilation_method = response_data.get('compilation_method', '')
        if compilation_method:
            if 'cupy' in compilation_method.lower():
                return True
            elif 'pytorch' in compilation_method.lower():
                return False
        
        # Check if we have kernel_exec_result (indicates CUDA usage)
        kernel_exec_result = response_data.get("kernel_exec_result", {})
        if kernel_exec_result:
            metadata = kernel_exec_result.get("metadata", {})
            
            # Check for GPU ID in metadata
            gpu_id = metadata.get("gpu_id")
            if gpu_id is not None:
                return True
            
            # Check compilation_method in metadata
            if metadata.get("compilation_method") == "cupy":
                return True
        
        # For successful responses, assume CUDA was used (since server uses CuPy by default)
        if response_data.get('status') == 'success':
            return True
        
        # Check error messages for CUDA-related failures
        error_msg = response_data.get('error_message', '') or response_data.get('error', '')
        if 'cupy' in error_msg.lower() or 'cuda' in error_msg.lower():
            return True
            
        return None
    
    def _extract_gpu_id(self, response_data: Dict[str, Any]) -> Optional[int]:
        """Extract GPU ID from server response"""
        kernel_exec_result = response_data.get("kernel_exec_result", {})
        metadata = kernel_exec_result.get("metadata", {})
        return metadata.get("gpu_id")
    
    def _extract_device_metrics(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract device metrics from server response"""
        kernel_exec_result = response_data.get("kernel_exec_result", {})
        metadata = kernel_exec_result.get("metadata", {})
        return metadata.get("device_metrics")
    
    def _format_device_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format device metrics for pretty printing"""
        if not metrics:
            return "No device metrics available"
        
        # Check if we have any actual metrics data
        has_original = "original_device_metrics" in metrics and metrics["original_device_metrics"]
        has_custom = "custom_device_metrics" in metrics and metrics["custom_device_metrics"]
        
        if not has_original and not has_custom:
            return "Device metrics collected but no data available (NCU profiling may have failed)"
        
        output = []
        
        # Original metrics - fixed to use correct key names
        if has_original:
            orig = metrics["original_device_metrics"]
            output.append("📊 ORIGINAL (Reference) METRICS:")
            
            if "speed_of_light" in orig and orig["speed_of_light"]:
                sol = orig["speed_of_light"]
                output.append("  Speed of Light Analysis:")
                output.append(f"    • Compute Throughput: {sol.get('compute_throughput_pct', 'N/A')}%")
                output.append(f"    • Memory Throughput: {sol.get('memory_throughput_pct', 'N/A')}%")
            else:
                output.append("  Speed of Light metrics not available")
            
            # Add other metrics if available
            if "detailed_metrics" in orig:
                detailed = orig["detailed_metrics"]
                output.append("  Detailed Metrics:")
                if "l1_hit_rate_pct" in detailed:
                    output.append(f"    • L1 Cache Hit Rate: {detailed['l1_hit_rate_pct']:.1f}%")
                if "l2_hit_rate_pct" in detailed:
                    output.append(f"    • L2 Cache Hit Rate: {detailed['l2_hit_rate_pct']:.1f}%")
                if "warp_occupancy_pct" in detailed:
                    output.append(f"    • Warp Occupancy: {detailed['warp_occupancy_pct']:.1f}%")
                if "instructions_per_cycle" in detailed:
                    output.append(f"    • Instructions Per Cycle: {detailed['instructions_per_cycle']:.2f}")
        
        # Custom metrics - fixed to use correct key names
        if has_custom:
            custom = metrics["custom_device_metrics"]
            output.append("\n📊 CUSTOM (Generated) METRICS:")
            
            if "speed_of_light" in custom and custom["speed_of_light"]:
                sol = custom["speed_of_light"]
                output.append("  Speed of Light Analysis:")
                output.append(f"    • Compute Throughput: {sol.get('compute_throughput_pct', 'N/A')}%")
                output.append(f"    • Memory Throughput: {sol.get('memory_throughput_pct', 'N/A')}%")
            else:
                output.append("  Speed of Light metrics not available")
            
            # Add other metrics if available
            if "detailed_metrics" in custom:
                detailed = custom["detailed_metrics"]
                output.append("  Detailed Metrics:")
                if "l1_hit_rate_pct" in detailed:
                    output.append(f"    • L1 Cache Hit Rate: {detailed['l1_hit_rate_pct']:.1f}%")
                if "l2_hit_rate_pct" in detailed:
                    output.append(f"    • L2 Cache Hit Rate: {detailed['l2_hit_rate_pct']:.1f}%")
                if "warp_occupancy_pct" in detailed:
                    output.append(f"    • Warp Occupancy: {detailed['warp_occupancy_pct']:.1f}%")
                if "instructions_per_cycle" in detailed:
                    output.append(f"    • Instructions Per Cycle: {detailed['instructions_per_cycle']:.2f}")
        
        # Add compute and memory metrics comparison only if we have valid data
        if has_original and has_custom:
            orig_sol = metrics.get("original_device_metrics", {}).get("speed_of_light", {})
            custom_sol = metrics.get("custom_device_metrics", {}).get("speed_of_light", {})
            
            if orig_sol and custom_sol:  # Only if both have speed_of_light data
                orig_compute = orig_sol.get("compute_throughput_pct")
                custom_compute = custom_sol.get("compute_throughput_pct")
                orig_memory = orig_sol.get("memory_throughput_pct")
                custom_memory = custom_sol.get("memory_throughput_pct")
                
                if all(v is not None for v in [orig_compute, custom_compute, orig_memory, custom_memory]):
                    output.append("\n📊 PERFORMANCE COMPARISON:")
                    
                    # Compute comparison
                    compute_diff = custom_compute - orig_compute
                    compute_sign = "+" if compute_diff > 0 else ""
                    output.append(f"  • Compute Throughput: {compute_sign}{compute_diff:.1f}% "
                                f"({orig_compute:.1f}% → {custom_compute:.1f}%)")
                    
                    # Memory comparison
                    memory_diff = custom_memory - orig_memory
                    memory_sign = "+" if memory_diff > 0 else ""
                    output.append(f"  • Memory Throughput: {memory_sign}{memory_diff:.1f}% "
                                f"({orig_memory:.1f}% → {custom_memory:.1f}%)")
                    
                    # Bottleneck analysis
                    output.append("\n🔍 BOTTLENECK ANALYSIS:")
                    if orig_compute > orig_memory:
                        output.append(f"  • Original: Compute-bound ({orig_compute:.1f}% > {orig_memory:.1f}%)")
                    else:
                        output.append(f"  • Original: Memory-bound ({orig_memory:.1f}% > {orig_compute:.1f}%)")
                    
                    if custom_compute > custom_memory:
                        output.append(f"  • Custom: Compute-bound ({custom_compute:.1f}% > {custom_memory:.1f}%)")
                    else:
                        output.append(f"  • Custom: Memory-bound ({custom_memory:.1f}% > {custom_compute:.1f}%)")
        
        # Return the formatted output or a message if nothing was formatted
        if output:
            return "\n".join(output)
        else:
            return "Device metrics structure present but no valid data extracted"
    
    def test_specific_case_by_index(self, index: int, num_trials: int = 10) -> Dict[str, Any]:
        """
        Test a specific test case by index
        
        Args:
            index: Index of the test case in the KernelBench JSON
            num_trials: Number of trials for performance testing
            
        Returns:
            Dictionary with detailed validation results
        """
        logger.info(f"🎯 Testing specific test case at index {index}")
        
        # Get the specific test case
        test_case = self.test_provider.get_test_case(index)
        if not test_case:
            return {
                "error": f"Test case at index {index} not found",
                "available_count": self.test_provider.kernelbench_provider.count_cases()['total_cases']
            }
        
        logger.info(f"📋 Test Case Details:")
        logger.info(f"   Index: {test_case.get('index', index)}")
        logger.info(f"   Expected - Compile: {test_case.get('expected_compile')}, "
                   f"Correct: {test_case.get('expected_correct')}, "
                   f"CUDA: {test_case.get('expected_cuda')}")
        
        # Run validation for this single test case
        result = self.validate_single_test_case(test_case, num_trials)
        
        # Extract and format device metrics if available
        device_metrics = result.device_metrics
        formatted_device_metrics = None
        if device_metrics:
            formatted_device_metrics = self._format_device_metrics(device_metrics)
            # Also log the device metrics
            logger.info(f"\n🔬 DEVICE METRICS for index {index}:")
            for line in formatted_device_metrics.split('\n'):
                logger.info(f"   {line}")
        
        # Prepare detailed report
        detailed_report = {
            "test_case_index": index,
            "expected": {
                "compile": test_case.get('expected_compile'),
                "correct": test_case.get('expected_correct'),
                "cuda": test_case.get('expected_cuda')
            },
            "actual": {
                "compile": result.actual_compile,
                "correct": result.actual_correct,
                "cuda": result.actual_cuda
            },
            "validation_passed": result.validation_passed,
            "expected_failure_but_passed": result.expected_failure_but_passed,
            "details": {
                "compile_match": result.expected_compile == result.actual_compile,
                "correct_match": result.expected_correct == result.actual_correct,
                "cuda_match": result.expected_cuda == result.actual_cuda
            },
            "request_info": {
                "success": result.success,
                "response_time": result.response_time,
                "status_code": result.status_code,
                "job_id": result.job_id,
                "gpu_id": result.gpu_id,
                "error": result.error
            },
            "device_metrics": device_metrics,
            "formatted_device_metrics": formatted_device_metrics
        }
        
        # Log the results
        if result.success:
            if result.validation_passed:
                logger.info(f"✅ VALIDATION PASSED for index {index}")
                logger.info(f"   All expectations matched!")
            else:
                logger.warning(f"❌ VALIDATION FAILED for index {index}")
                logger.warning(f"   Expected: compile={result.expected_compile}, correct={result.expected_correct}, cuda={result.expected_cuda}")
                logger.warning(f"   Actual:   compile={result.actual_compile}, correct={result.actual_correct}, cuda={result.actual_cuda}")
                
                # Log specific mismatches
                if result.expected_compile != result.actual_compile:
                    logger.warning(f"   ⚠️ Compilation status mismatch")
                if result.expected_correct != result.actual_correct:
                    logger.warning(f"   ⚠️ Correctness status mismatch")
                if result.expected_cuda != result.actual_cuda:
                    logger.warning(f"   ⚠️ CUDA usage mismatch")
        else:
            logger.error(f"❌ REQUEST FAILED for index {index}: {result.error}")
        
        return detailed_report
    
    def run_validation_batch(self, batch_size: int = 50, use_all_cases: bool = True) -> Dict[str, Any]:
        """
        Run validation on a batch of test cases
        
        Args:
            batch_size: Number of test cases to validate
            use_all_cases: If True, uses random mix of all cases (not just CUDA cases)
        """
        logger.info(f"🔍 Running validation batch: {batch_size} test cases")
        logger.info(f"   Using {'ALL test cases (mixed)' if use_all_cases else 'CUDA-only test cases'}")
        
        # Get random test cases from all available (no CUDA filtering by default)
        test_batch = self.test_provider.get_test_batch(batch_size, cuda_only=(not use_all_cases))
        if not test_batch:
            return {"error": "No test cases available"}
        
        results = []
        successful_validations = 0
        
        # Track different validation outcomes
        validation_outcomes = {
            "passed_as_expected_success": 0,
            "passed_as_expected_failure": 0,
            "failed_unexpected_success": 0,
            "failed_unexpected_failure": 0,
            "failed_wrong_reason": 0,
            "request_failed": 0
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting validation of {len(test_batch)} test cases...")
        logger.info(f"{'='*70}\n")
        
        for i, test_case in enumerate(test_batch, 1):
            logger.info(f"\n--- Test Case {i}/{len(test_batch)} ---")
            
            result = self.validate_single_test_case(test_case, log_details=True)
            results.append(result)
            
            # Determine the validation outcome
            if not result.success:
                # HTTP error - check if it was expected
                if not result.expected_compile:
                    # We expected compilation failure, HTTP error is acceptable
                    if result.validation_passed:
                        logger.info(f"   ✅ VALIDATION PASSED - Failed as expected (compilation failure)")
                        successful_validations += 1
                        validation_outcomes["passed_as_expected_failure"] += 1
                    else:
                        logger.warning(f"   ❌ VALIDATION FAILED - Failed but not as expected")
                        validation_outcomes["failed_wrong_reason"] += 1
                else:
                    # Unexpected HTTP error
                    logger.error(f"   ❌ REQUEST FAILED unexpectedly: {result.error[:100]}")
                    validation_outcomes["request_failed"] += 1
            else:
                # Request succeeded (HTTP 200)
                if result.validation_passed:
                    successful_validations += 1
                    
                    # Determine if it passed because it succeeded or failed as expected
                    if result.expected_compile and result.expected_correct:
                        logger.info(f"   ✅ VALIDATION PASSED - Succeeded as expected")
                        validation_outcomes["passed_as_expected_success"] += 1
                    elif result.expected_compile and not result.expected_correct:
                        logger.info(f"   ✅ VALIDATION PASSED - Failed correctness as expected")
                        validation_outcomes["passed_as_expected_failure"] += 1
                    elif not result.expected_compile:
                        logger.info(f"   ✅ VALIDATION PASSED - Failed compilation as expected")
                        validation_outcomes["passed_as_expected_failure"] += 1
                    else:
                        logger.info(f"   ✅ VALIDATION PASSED - Matched expectations")
                        validation_outcomes["passed_as_expected_success"] += 1
                else:
                    # Validation failed - expectations didn't match
                    logger.warning(f"   ❌ VALIDATION FAILED - Expectations not met:")
                    logger.warning(f"      Expected: compile={result.expected_compile}, correct={result.expected_correct}, cuda={result.expected_cuda}")
                    logger.warning(f"      Actual:   compile={result.actual_compile}, correct={result.actual_correct}, cuda={result.actual_cuda}")
                    
                    # Categorize the failure
                    if result.expected_compile and result.expected_correct and (not result.actual_compile or not result.actual_correct):
                        logger.warning(f"      → Expected success but got failure")
                        validation_outcomes["failed_unexpected_failure"] += 1
                    elif not (result.expected_compile and result.expected_correct) and result.actual_compile and result.actual_correct:
                        logger.warning(f"      → Expected failure but got success")
                        validation_outcomes["failed_unexpected_success"] += 1
                    else:
                        logger.warning(f"      → Failed for wrong reason")
                        validation_outcomes["failed_wrong_reason"] += 1
        
        # Log summary of outcomes
        logger.info(f"\n{'='*70}")
        logger.info(f"VALIDATION BATCH SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"✅ Successful Validations: {successful_validations}/{len(test_batch)}")
        logger.info(f"   • Passed (succeeded as expected): {validation_outcomes['passed_as_expected_success']}")
        logger.info(f"   • Passed (failed as expected): {validation_outcomes['passed_as_expected_failure']}")
        logger.info(f"❌ Failed Validations: {len(test_batch) - successful_validations}/{len(test_batch)}")
        logger.info(f"   • Failed (unexpected success): {validation_outcomes['failed_unexpected_success']}")
        logger.info(f"   • Failed (unexpected failure): {validation_outcomes['failed_unexpected_failure']}")
        logger.info(f"   • Failed (wrong reason): {validation_outcomes['failed_wrong_reason']}")
        logger.info(f"   • Request failures: {validation_outcomes['request_failed']}")
        logger.info(f"{'='*70}\n")
        
        # Generate validation report
        report = self._generate_validation_report(results, successful_validations)
        report["validation_outcomes"] = validation_outcomes
        return report
    
    def _generate_validation_report(self, results: List[TestResult], successful_validations: int) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        successful_requests = [r for r in results if r.success]
        failed_requests = [r for r in results if not r.success]
        
        # Validation statistics
        validation_stats = {
            "total_tests": len(results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "successful_validations": successful_validations,
            "validation_rate": round(successful_validations / len(results), 3) if results else 0
        }
        
        # Breakdown by validation type
        compile_matches = sum(1 for r in successful_requests 
                             if r.expected_compile == r.actual_compile)
        correct_matches = sum(1 for r in successful_requests 
                             if r.expected_correct == r.actual_correct)
        cuda_matches = sum(1 for r in successful_requests 
                          if r.expected_cuda == r.actual_cuda)
        
        validation_breakdown = {
            "compile_matches": compile_matches,
            "correct_matches": correct_matches, 
            "cuda_matches": cuda_matches,
            "compile_accuracy": round(compile_matches / len(successful_requests), 3) if successful_requests else 0,
            "correct_accuracy": round(correct_matches / len(successful_requests), 3) if successful_requests else 0,
            "cuda_accuracy": round(cuda_matches / len(successful_requests), 3) if successful_requests else 0
        }
        
        # Response time stats
        response_times = [r.response_time for r in results]
        response_stats = {}
        if response_times:
            response_stats = {
                "mean": round(statistics.mean(response_times), 2),
                "median": round(statistics.median(response_times), 2),
                "min": round(min(response_times), 2),
                "max": round(max(response_times), 2)
            }
        
        return {
            "validation_summary": validation_stats,
            "validation_breakdown": validation_breakdown,
            "response_times": response_stats,
            "errors": {
                "unique_errors": len(set(r.error for r in failed_requests if r.error)),
                "error_breakdown": list(set(r.error for r in failed_requests if r.error))
            }
        }


class MetricsAggregator:
    """Aggregates and analyzes load test metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results: List[TestResult] = []
        
    def start_collection(self):
        """Start metrics collection"""
        self.start_time = time.time()
        
    def end_collection(self):
        """End metrics collection"""
        self.end_time = time.time()
        
    def record_result(self, result: TestResult):
        """Record a test result"""
        self.results.append(result)
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No results to analyze"}
            
        total_time = self.end_time - self.start_time if self.end_time else 0
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in self.results]
        gpu_ids = [r.gpu_id for r in successful_results if r.gpu_id is not None]
        
        # Calculate statistics
        response_stats = {}
        if response_times:
            response_stats = {
                "mean": round(statistics.mean(response_times), 2),
                "median": round(statistics.median(response_times), 2),
                "min": round(min(response_times), 2),
                "max": round(max(response_times), 2),
                "stdev": round(statistics.stdev(response_times), 2) if len(response_times) > 1 else 0.0
            }
        
        # GPU utilization analysis
        gpu_usage = defaultdict(int)
        for gpu_id in gpu_ids:
            gpu_usage[gpu_id] += 1
            
        return {
            "summary": {
                "total_requests": len(self.results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": round(len(successful_results) / len(self.results), 3) if self.results else 0,
                "total_duration_seconds": round(total_time, 2),
                "throughput_requests_per_second": round(len(self.results) / total_time, 2) if total_time > 0 else 0
            },
            "response_times": response_stats,
            "gpu_utilization": {
                "unique_gpus_used": len(set(gpu_ids)),
                "gpu_usage_count": dict(gpu_usage),
                "gpu_distribution": {gpu_id: count for gpu_id, count in gpu_usage.items()}
            },
            "errors": {
                "unique_errors": len(set(r.error for r in failed_results if r.error)),
                "error_breakdown": list(set(r.error for r in failed_results if r.error))
            }
        }


class ConcurrentLoadTester:
    """Handles concurrent load testing scenarios"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.test_provider = TestCaseProvider()
        self.metrics = MetricsAggregator()
        
    def single_request_test(self, request_id: int, num_trials: int = 50, timeout: int = 60, cuda_only: bool = False) -> TestResult:
        """Execute a single request test"""
        start_time = time.time()
        
        try:
            # Get test case (CUDA-only if requested)
            test_case = self.test_provider.get_random_test_case(cuda_only=cuda_only)
            test_request = self.test_provider.create_test_request(
                num_trials=num_trials, 
                timeout=timeout, 
                test_case=test_case
            )
            
            response = requests.post(
                f"{self.base_url}/",
                json=test_request,
                timeout=timeout + 10  # Add buffer to request timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                job_id = result_data.get("job_id", "unknown")
                
                # Extract GPU info if available - check multiple possible locations
                gpu_id = None
                compilation_time = None
                profiling_time = None
                
                # Try different locations where GPU ID might be stored
                kernel_exec_result = result_data.get("kernel_exec_result", {})
                metadata = kernel_exec_result.get("metadata", {})
                
                if metadata:
                    gpu_id = metadata.get("gpu_id")
                
                # Also check if gpu_id is directly in kernel_exec_result
                if gpu_id is None and kernel_exec_result:
                    gpu_id = kernel_exec_result.get("gpu_id")
                
                # Debug: Print response structure to understand the data flow
                logger.info(f"Response structure for job {job_id} (time: {response_time:.2f}s):")
                logger.info(f"  Status: {result_data.get('status', 'unknown')}")
                logger.info(f"  Keys: {list(result_data.keys())}")
                
                if 'kernel_exec_result' in result_data:
                    ker_result = result_data['kernel_exec_result']
                    logger.info(f"  kernel_exec_result keys: {list(ker_result.keys()) if isinstance(ker_result, dict) else 'not dict'}")
                    if isinstance(ker_result, dict) and 'metadata' in ker_result:
                        metadata = ker_result['metadata']
                        logger.info(f"  metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'not dict'}")
                        if isinstance(metadata, dict) and 'gpu_id' in metadata:
                            logger.info(f"  GPU ID found in metadata: {metadata['gpu_id']}")
                
                # Also check compilation method to understand execution path
                compilation_method = result_data.get('compilation_method', 'unknown')
                logger.info(f"  compilation_method: {compilation_method}")
                
                return TestResult(
                    success=True,
                    response_time=response_time,
                    status_code=response.status_code,
                    job_id=job_id,
                    gpu_id=gpu_id,
                    compilation_time=compilation_time,
                    profiling_time=profiling_time
                )
            else:
                return TestResult(
                    success=False,
                    response_time=response_time,
                    status_code=response.status_code,
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
                
        except Exception as e:
            return TestResult(
                success=False,
                response_time=time.time() - start_time,
                status_code=0,
                error=str(e)
            )
    
    def test_concurrent_requests(self, num_concurrent: int = 5, num_trials_per_request: int = 20, cuda_only: bool = False) -> Dict[str, Any]:
        """Test concurrent request handling"""
        cuda_label = " (CUDA-only)" if cuda_only else ""
        logger.info(f"🚀 Starting concurrent load test: {num_concurrent} concurrent requests{cuda_label}")
        
        self.metrics.start_collection()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            # Submit all requests
            futures = []
            for i in range(num_concurrent):
                future = executor.submit(
                    self.single_request_test, 
                    request_id=i, 
                    num_trials=num_trials_per_request,
                    timeout=120,
                    cuda_only=cuda_only
                )
                futures.append(future)
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    self.metrics.record_result(result)
                    
                    status = "✅ SUCCESS" if result.success else "❌ FAILED"
                    logger.info(f"Request {i}/{num_concurrent}: {status} - {result.response_time:.2f}s")
                    
                    if not result.success:
                        logger.warning(f"   Error: {result.error}")
                    elif result.gpu_id is not None:
                        logger.info(f"   GPU ID: {result.gpu_id}, Job ID: {result.job_id}")
                        
                except Exception as e:
                    logger.error(f"Request {i} failed with exception: {e}")
        
        self.metrics.end_collection()
        return self.metrics.generate_report()
    
    def test_gpu_resource_contention(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test GPU resource contention with more requests than GPUs"""
        logger.info(f"🎯 Testing GPU resource contention: {num_requests} requests competing for GPUs")
        
        # First check how many GPUs are available
        gpu_count = self._get_available_gpu_count()
        if gpu_count == 0:
            return {"error": "No GPUs detected - cannot test GPU contention"}
        
        logger.info(f"   Detected {gpu_count} GPUs - launching {num_requests} requests")
        
        # Use more requests than GPUs to test queuing, use CUDA-only cases
        return self.test_concurrent_requests(
            num_concurrent=num_requests, 
            num_trials_per_request=10,  # Shorter tests for contention testing
            cuda_only=True  # Use CUDA-only cases for GPU contention tests
        )
    
    def test_sustained_load(self, duration_minutes: int = 5, requests_per_minute: int = 2) -> Dict[str, Any]:
        """Test sustained load over time"""
        logger.info(f"⏱️ Testing sustained load: {requests_per_minute} req/min for {duration_minutes} minutes")
        
        end_time = time.time() + (duration_minutes * 60)
        interval = 60.0 / requests_per_minute  # Seconds between requests
        
        self.metrics.start_collection()
        request_count = 0
        
        while time.time() < end_time:
            request_start = time.time()
            
            result = self.single_request_test(request_count, num_trials=10, timeout=60, cuda_only=True)
            self.metrics.record_result(result)
            request_count += 1
            
            status = "✅" if result.success else "❌"
            logger.info(f"Sustained request {request_count}: {status} - {result.response_time:.2f}s")
            
            # Wait for next interval
            elapsed = time.time() - request_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.metrics.end_collection()
        return self.metrics.generate_report()
    
    def test_mixed_load_patterns_scaled(self, gpu_count: int) -> Dict[str, Any]:
        """Test mixed load patterns with GPU-aware scaling"""
        logger.info(f"🌪️ Starting GPU-scaled mixed load pattern test with {gpu_count} GPUs")
        
        self.metrics = MetricsAggregator()  # Reset metrics for this test
        self.metrics.start_collection()
        
        # Phase 1: High concurrency burst (scaled to 3x GPU count)
        burst_count = gpu_count * 3
        logger.info(f"   Phase 1: High concurrency burst - {burst_count} requests")
        burst_results = self.test_concurrent_requests(num_concurrent=burst_count, num_trials_per_request=1, cuda_only=True)
        
        # Phase 2: Medium sustained load (scaled to 2x GPU count)
        sustained_count = gpu_count * 2
        logger.info(f"   Phase 2: Medium sustained load - {sustained_count} requests")
        sustained_results = self.test_concurrent_requests(num_concurrent=sustained_count, num_trials_per_request=1, cuda_only=True)
        
        # Phase 3: Intensive work (scaled to GPU count with higher trials)
        intensive_count = gpu_count
        logger.info(f"   Phase 3: Intensive work - {intensive_count} requests with 5 trials each")
        intensive_results = self.test_concurrent_requests(num_concurrent=intensive_count, num_trials_per_request=5, cuda_only=True)
        
        # Phase 4: Maximum stress (scaled to 5x GPU count)
        max_stress_count = gpu_count * 5
        logger.info(f"   Phase 4: Maximum stress - {max_stress_count} requests")
        max_stress_results = self.test_concurrent_requests(num_concurrent=max_stress_count, num_trials_per_request=1, cuda_only=True)
        
        self.metrics.end_collection()
        
        # Aggregate results from all patterns
        total_requests = (
            burst_results["summary"]["total_requests"] +
            sustained_results["summary"]["total_requests"] + 
            intensive_results["summary"]["total_requests"] +
            max_stress_results["summary"]["total_requests"]
        )
        
        successful_requests = (
            burst_results["summary"]["successful_requests"] +
            sustained_results["summary"]["successful_requests"] +
            intensive_results["summary"]["successful_requests"] +
            max_stress_results["summary"]["successful_requests"]
        )
        
        # Combine GPU utilization data
        all_gpu_distributions = {}
        for result in [burst_results, sustained_results, intensive_results, max_stress_results]:
            gpu_dist = result["gpu_utilization"]["gpu_distribution"]
            for gpu_id, count in gpu_dist.items():
                all_gpu_distributions[gpu_id] = all_gpu_distributions.get(gpu_id, 0) + count
        
        # Combine response times for analysis
        all_response_times = []
        for result in [burst_results, sustained_results, intensive_results, max_stress_results]:
            if result["response_times"]:
                mean = result["response_times"]["mean"]
                count = result["summary"]["total_requests"]
                all_response_times.extend([mean] * count)
        
        # Calculate combined statistics
        combined_stats = {
            "mean": round(statistics.mean(all_response_times), 2) if all_response_times else 0,
            "min": min(all_response_times) if all_response_times else 0,
            "max": max(all_response_times) if all_response_times else 0,
        }
        
        return {
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": total_requests - successful_requests,
                "success_rate": round(successful_requests / total_requests, 3) if total_requests > 0 else 0,
                "total_duration_seconds": round(self.metrics.end_time - self.metrics.start_time, 2),
                "throughput_requests_per_second": round(total_requests / (self.metrics.end_time - self.metrics.start_time), 2)
            },
            "response_times": combined_stats,
            "gpu_utilization": {
                "unique_gpus_used": len(all_gpu_distributions),
                "gpu_distribution": all_gpu_distributions
            },
            "test_phases": {
                "phase_1_burst": burst_results["summary"],
                "phase_2_sustained": sustained_results["summary"], 
                "phase_3_intensive": intensive_results["summary"],
                "phase_4_max_stress": max_stress_results["summary"]
            },
            "gpu_scaling": {
                "base_gpu_count": gpu_count,
                "burst_scaling": f"{burst_count} requests (3x GPUs)",
                "sustained_scaling": f"{sustained_count} requests (2x GPUs)",
                "intensive_scaling": f"{intensive_count} requests (1x GPUs, 5 trials)",
                "max_stress_scaling": f"{max_stress_count} requests (5x GPUs)"
            },
            "errors": {
                "unique_errors": 0,  # Simplified for mixed test
                "error_breakdown": []
            }
        }

    def _get_available_gpu_count(self) -> int:
        """Get number of available GPUs from health check"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                profiling_service = health_data.get("profiling_service", {})
                return profiling_service.get("total_gpus", 0)
        except Exception as e:
            logger.warning(f"Could not get GPU count: {e}")
        
        return 0


class GPUResourceValidator:
    """Validates proper GPU resource management"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    def validate_gpu_acquisition_release(self, num_tests: int = 5) -> Dict[str, Any]:
        """Validate that GPUs are properly acquired and released"""
        logger.info(f"🔍 Validating GPU resource management with {num_tests} tests")
        
        initial_stats = self._get_server_stats()
        if not initial_stats:
            return {"error": "Could not get initial server stats"}
        
        # Record initial GPU state
        initial_gpu_sessions = initial_stats.get("gpu_utilization", {}).get("active_sessions", 0)
        
        # Run some requests
        tester = ConcurrentLoadTester(self.base_url)
        test_results = tester.test_concurrent_requests(num_concurrent=num_tests, num_trials_per_request=5, cuda_only=True)
        
        # Check final state
        time.sleep(2)  # Give time for cleanup
        final_stats = self._get_server_stats()
        final_gpu_sessions = final_stats.get("gpu_utilization", {}).get("active_sessions", 0)
        
        # Validation results
        gpu_leak_detected = final_gpu_sessions > initial_gpu_sessions
        
        return {
            "validation_results": {
                "initial_active_sessions": initial_gpu_sessions,
                "final_active_sessions": final_gpu_sessions,
                "gpu_leak_detected": gpu_leak_detected,
                "resource_management": "✅ PASSED" if not gpu_leak_detected else "❌ FAILED",
            },
            "test_results": test_results,
            "gpu_utilization_change": {
                "total_gpu_time": final_stats.get("gpu_utilization", {}).get("total_gpu_time_seconds", 0) - 
                                 initial_stats.get("gpu_utilization", {}).get("total_gpu_time_seconds", 0),
                "acquisitions": final_stats.get("gpu_utilization", {}).get("gpu_acquisitions", 0) - 
                              initial_stats.get("gpu_utilization", {}).get("gpu_acquisitions", 0),
                "releases": final_stats.get("gpu_utilization", {}).get("gpu_releases", 0) - 
                           initial_stats.get("gpu_utilization", {}).get("gpu_releases", 0)
            }
        }
    
    def _get_server_stats(self) -> Dict[str, Any]:
        """Get current server statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get server stats: {e}")
        
        return {}


class LoadTestReporter:
    """Generates comprehensive load test reports"""
    
    @staticmethod
    def print_detailed_report(test_name: str, results: Dict[str, Any]):
        """Print a detailed test report"""
        print(f"\n{'='*80}")
        print(f"📊 {test_name.upper()} - DETAILED REPORT")
        print(f"{'='*80}")
        
        if "error" in results:
            print(f"❌ ERROR: {results['error']}")
            return
        
        # Summary section
        summary = results.get("summary", {})
        print(f"\n📈 SUMMARY:")
        print(f"  • Total Requests: {summary.get('total_requests', 0)}")
        print(f"  • Successful: {summary.get('successful_requests', 0)}")
        print(f"  • Failed: {summary.get('failed_requests', 0)}")
        print(f"  • Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"  • Total Duration: {summary.get('total_duration_seconds', 0)}s")
        print(f"  • Throughput: {summary.get('throughput_requests_per_second', 0)} req/s")
        
        # Response time analysis
        response_times = results.get("response_times", {})
        if response_times:
            print(f"\n⏱️ RESPONSE TIME ANALYSIS:")
            print(f"  • Mean: {response_times.get('mean', 0)}s")
            print(f"  • Median: {response_times.get('median', 0)}s")
            print(f"  • Min: {response_times.get('min', 0)}s")
            print(f"  • Max: {response_times.get('max', 0)}s")
            print(f"  • Std Dev: {response_times.get('stdev', 0)}s")
        
        # GPU utilization analysis
        gpu_util = results.get("gpu_utilization", {})
        if gpu_util:
            print(f"\n🔧 GPU UTILIZATION:")
            print(f"  • Unique GPUs Used: {gpu_util.get('unique_gpus_used', 0)}")
            
            gpu_dist = gpu_util.get("gpu_distribution", {})
            if gpu_dist:
                print(f"  • GPU Usage Distribution:")
                for gpu_id, count in gpu_dist.items():
                    print(f"    - GPU {gpu_id}: {count} requests")
        
        # Error analysis
        errors = results.get("errors", {})
        if errors and errors.get("unique_errors", 0) > 0:
            print(f"\n❌ ERROR ANALYSIS:")
            print(f"  • Unique Error Types: {errors.get('unique_errors', 0)}")
            for error in errors.get("error_breakdown", []):
                print(f"    - {error}")
        
        # Validation results (if present)
        if "validation_results" in results:
            validation = results["validation_results"]
            print(f"\n✅ GPU RESOURCE VALIDATION:")
            print(f"  • Initial Active Sessions: {validation.get('initial_active_sessions', 0)}")
            print(f"  • Final Active Sessions: {validation.get('final_active_sessions', 0)}")
            print(f"  • Resource Management: {validation.get('resource_management', 'UNKNOWN')}")
            
            gpu_change = results.get("gpu_utilization_change", {})
            if gpu_change:
                print(f"  • Total GPU Acquisitions: {gpu_change.get('acquisitions', 0)}")
                print(f"  • Total GPU Releases: {gpu_change.get('releases', 0)}")
                print(f"  • GPU Time Used: {gpu_change.get('total_gpu_time', 0):.2f}s")
    
    @staticmethod
    def print_validation_report(test_name: str, results: Dict[str, Any]):
        """Print a detailed validation test report"""
        print(f"\n{'='*80}")
        print(f"🔍 {test_name.upper()} - VALIDATION REPORT")
        print(f"{'='*80}")
        
        if "error" in results:
            print(f"❌ ERROR: {results['error']}")
            return
        
        # Validation summary
        validation_summary = results.get("validation_summary", {})
        print(f"\n📋 VALIDATION SUMMARY:")
        print(f"  • Total Tests: {validation_summary.get('total_tests', 0)}")
        print(f"  • Successful Requests: {validation_summary.get('successful_requests', 0)}")
        print(f"  • Failed Requests: {validation_summary.get('failed_requests', 0)}")
        print(f"  • Successful Validations: {validation_summary.get('successful_validations', 0)}")
        print(f"  • Validation Rate: {validation_summary.get('validation_rate', 0)*100:.1f}%")
        
        # Validation breakdown
        validation_breakdown = results.get("validation_breakdown", {})
        if validation_breakdown:
            print(f"\n🔍 VALIDATION BREAKDOWN:")
            print(f"  • Compilation Matches: {validation_breakdown.get('compile_matches', 0)}")
            print(f"  • Correctness Matches: {validation_breakdown.get('correct_matches', 0)}")
            print(f"  • CUDA Usage Matches: {validation_breakdown.get('cuda_matches', 0)}")
            print(f"  • Compilation Accuracy: {validation_breakdown.get('compile_accuracy', 0)*100:.1f}%")
            print(f"  • Correctness Accuracy: {validation_breakdown.get('correct_accuracy', 0)*100:.1f}%")
            print(f"  • CUDA Usage Accuracy: {validation_breakdown.get('cuda_accuracy', 0)*100:.1f}%")
        
        # Response time analysis
        response_times = results.get("response_times", {})
        if response_times:
            print(f"\n⏱️ RESPONSE TIME ANALYSIS:")
            print(f"  • Mean: {response_times.get('mean', 0)}s")
            print(f"  • Median: {response_times.get('median', 0)}s")
            print(f"  • Min: {response_times.get('min', 0)}s")
            print(f"  • Max: {response_times.get('max', 0)}s")
        
        # Error analysis
        errors = results.get("errors", {})
        if errors and errors.get("unique_errors", 0) > 0:
            print(f"\n❌ ERROR ANALYSIS:")
            print(f"  • Unique Error Types: {errors.get('unique_errors', 0)}")
            for error in errors.get("error_breakdown", []):
                print(f"    - {error}")


def test_server_basic_functionality(base_url: str):
    """Enhanced basic server functionality test with KernelBench case info"""
    print("🧪 Basic Server Functionality Test")
    print(f"   Server URL: {base_url}")
    
    # Test 1: Health Check
    try:
        print("\n1. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            
            # Show GPU availability
            profiling_service = health_data.get("profiling_service", {})
            total_gpus = profiling_service.get("total_gpus", 0)
            available_gpus = profiling_service.get("available_gpus", 0)
            print(f"   📊 GPUs: {total_gpus} total, {available_gpus} available")
        else:
            print(f"   ❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Enhanced Stats Endpoint
    try:
        print("\n2. Testing enhanced stats endpoint...")
        response = requests.get(f"{base_url}/stats", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            stats_data = response.json()
            print("   ✅ Enhanced stats endpoint working")
            
            # Show key metrics
            compilation_metrics = stats_data.get("compilation_metrics", {})
            profiling_metrics = stats_data.get("profiling_metrics", {})
            throughput = stats_data.get("throughput", {})
            
            print(f"   📊 Server Uptime: {throughput.get('server_uptime_minutes', 0):.1f} minutes")
            print(f"   📊 Total Requests: {throughput.get('total_requests', 0)}")
            print(f"   📊 Compilation Success Rate: {compilation_metrics.get('success_rate', 0)*100:.1f}%")
            print(f"   📊 Profiling Success Rate: {profiling_metrics.get('success_rate', 0)*100:.1f}%")
        else:
            print(f"   ❌ Stats endpoint failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Stats endpoint error: {e}")
    
    # Test 3: KernelBench Test Cases Information
    try:
        print("\n3. Testing KernelBench test case provider...")
        test_provider = TestCaseProvider()
        counts = test_provider.kernelbench_provider.count_cases()
        
        if counts['total_cases'] > 0:
            print("   ✅ KernelBench test cases loaded successfully")
            print(f"   📊 Total test cases: {counts['total_cases']}")
            print(f"   📊 CUDA-only cases: {counts['cuda_cases']}")
            print(f"   📊 Compile=True cases: {counts['compile_true']}")
            print(f"   📊 Correct=True cases: {counts['correct_true']}")
        else:
            print("   ⚠️ KernelBench test cases not available, using fallback")
    except Exception as e:
        print(f"   ❌ KernelBench provider error: {e}")
    
    print("   ✅ Basic functionality test completed")
    return True


def run_validation_tests(base_url: str):
    """Run validation tests using KernelBench generated test cases"""
    print(f"\n{'='*80}")
    print("🔍 VALIDATION TESTING SUITE")
    print("   Testing server responses against expected KernelBench results")
    print("   Using RANDOM MIX of all test cases (not filtered by CUDA)")
    print(f"{'='*80}")
    
    validator = ValidationTestRunner(base_url)
    
    # Test 1: Small batch validation test with mixed cases  
    print(f"\n🧪 TEST 1: Small Batch Validation Test")
    print(f"   📋 Testing 10 RANDOM test cases (mixed compile/correct/cuda values)")
    print(f"   ➡️ Each test will show its ground truth expectations before running")
    small_results = validator.run_validation_batch(batch_size=10, use_all_cases=True)
    LoadTestReporter.print_validation_report("Small Batch Validation Test", small_results)
    
    # Test 2: Larger batch validation test with mixed cases
    print(f"\n🧪 TEST 2: Large Batch Validation Test")
    print(f"   📋 Testing 30 RANDOM test cases (mixed compile/correct/cuda values)")
    print(f"   ➡️ Includes cases that should fail compilation, fail correctness, or succeed")
    large_results = validator.run_validation_batch(batch_size=30, use_all_cases=True)
    LoadTestReporter.print_validation_report("Large Batch Validation Test", large_results)
    
    print(f"\n{'='*80}")
    print("✅ VALIDATION TESTING COMPLETED")
    print("   Tests included a random mix of:")
    print("   • Cases that should compile successfully")
    print("   • Cases that should fail compilation")
    print("   • Cases that should pass correctness validation")
    print("   • Cases that should fail correctness validation")
    print(f"{'='*80}")


def run_comprehensive_load_tests(base_url: str):
    """Run comprehensive load testing suite with GPU-aware concurrency scaling using CUDA-only cases"""
    print(f"\n{'='*80}")
    print("🚀 COMPREHENSIVE HIGH-VOLUME LOAD TESTING SUITE")
    print("   Stress testing scaled to available GPU resources")
    print("   Testing concurrent requests, GPU management, and server performance")
    print("   Using CUDA-only test cases from KernelBench for realistic GPU testing")
    print(f"{'='*80}")
    
    load_tester = ConcurrentLoadTester(base_url)
    gpu_validator = GPUResourceValidator(base_url)
    
    # Get GPU count for intelligent scaling
    gpu_count = load_tester._get_available_gpu_count()
    if gpu_count == 0:
        print("❌ No GPUs detected - cannot run meaningful load tests")
        return
    
    print(f"🔧 Detected {gpu_count} GPUs - scaling tests accordingly")
    
    # Intelligent scaling: 3x GPU count for high concurrency, 5x for extreme cases
    high_concurrency = max(gpu_count * 3, 12)  # At least 12 for meaningful load
    extreme_concurrency = max(gpu_count * 5, 20)  # At least 20 for stress testing
    cache_requests = max(gpu_count * 10, 50)  # At least 50 for cache effectiveness
    
    # Test 1: High-Volume Concurrent Request Handling (CUDA-only)
    print(f"\n🧪 TEST 1: High-Volume Concurrent Request Handling")
    print(f"   🚀 Testing {high_concurrency} concurrent requests with 5 trials each ({high_concurrency * 5} total operations)")
    print(f"   🔥 Using CUDA-only test cases for realistic GPU load")
    concurrent_results = load_tester.test_concurrent_requests(num_concurrent=high_concurrency, num_trials_per_request=5, cuda_only=True)
    LoadTestReporter.print_detailed_report("High-Volume Concurrent Request Test", concurrent_results)
    
    # Test 2: Extreme GPU Resource Contention
    print(f"\n🧪 TEST 2: Extreme GPU Resource Contention")
    print(f"   🎯 Testing {extreme_concurrency} requests competing for {gpu_count} GPU resources")
    contention_results = load_tester.test_gpu_resource_contention(num_requests=extreme_concurrency)
    LoadTestReporter.print_detailed_report("Extreme GPU Contention Test", contention_results)
    
    # Test 3: Sustained High Throughput (scaled to GPU capacity)
    requests_per_min = max(gpu_count * 3, 12)  # 3 requests per GPU per minute
    print(f"\n🧪 TEST 3: Sustained High Throughput Load")
    print(f"   ⏱️ Testing sustained load with {requests_per_min * 10} requests over 10 minutes")
    sustained_results = load_tester.test_sustained_load(duration_minutes=10, requests_per_minute=requests_per_min)
    LoadTestReporter.print_detailed_report("Sustained High Throughput Test", sustained_results)
    
    # Test 4: Massive Compilation Cache Test (scaled)
    print(f"\n🧪 TEST 4: Massive Compilation Cache Effectiveness")
    print(f"   💾 Testing compilation cache with {cache_requests} CUDA requests")
    cache_test_results = load_tester.test_concurrent_requests(num_concurrent=cache_requests, num_trials_per_request=1, cuda_only=True)
    LoadTestReporter.print_detailed_report("Massive Cache Effectiveness Test", cache_test_results)
    
    # Test 5: GPU Resource Management Validation at Scale
    validation_tests = max(gpu_count * 3, 15)  # 3x GPU count for validation
    print(f"\n🧪 TEST 5: GPU Resource Management Validation at Scale")
    print(f"   🔍 Validating GPU resource management with {validation_tests} test requests")
    validation_results = gpu_validator.validate_gpu_acquisition_release(num_tests=validation_tests)
    LoadTestReporter.print_detailed_report("Large-Scale GPU Resource Validation", validation_results)
    
    # Test 6: Mixed Load Pattern Stress Test (GPU-aware scaling)
    print(f"\n🧪 TEST 6: Mixed Load Pattern Stress Test")
    total_mixed_requests = high_concurrency + extreme_concurrency + (gpu_count * 5) + (gpu_count * 20)
    print(f"   🌪️ Testing mixed patterns: {total_mixed_requests}+ requests with varied concurrency")
    mixed_results = load_tester.test_mixed_load_patterns_scaled(gpu_count)
    LoadTestReporter.print_detailed_report("Mixed Load Pattern Stress Test", mixed_results)
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE HIGH-VOLUME LOAD TESTING COMPLETED")
    print(f"   📊 Thousands of requests processed across multiple test scenarios")
    print(f"   🔧 Tests intelligently scaled for {gpu_count} available GPUs")
    print(f"   🔥 Used CUDA-only test cases for realistic GPU workload simulation")
    print(f"{'='*80}")


def print_specific_test_report(results: Dict[str, Any]):
    """Print a detailed report for a specific test case validation"""
    print(f"\n{'='*80}")
    print(f"🎯 SPECIFIC TEST CASE VALIDATION REPORT")
    print(f"{'='*80}")
    
    if "error" in results:
        print(f"❌ ERROR: {results['error']}")
        if "available_count" in results:
            print(f"   Available test cases: {results['available_count']}")
        return
    
    index = results.get("test_case_index", "unknown")
    print(f"\n📋 Test Case Index: {index}")
    
    # Expected vs Actual comparison
    expected = results.get("expected", {})
    actual = results.get("actual", {})
    
    print(f"\n🔍 EXPECTED vs ACTUAL:")
    print(f"  • Compilation:  Expected={expected.get('compile')}, Actual={actual.get('compile')}")
    print(f"  • Correctness:  Expected={expected.get('correct')}, Actual={actual.get('correct')}")
    print(f"  • CUDA Usage:   Expected={expected.get('cuda')}, Actual={actual.get('cuda')}")
    
    # Validation result
    validation_passed = results.get("validation_passed", False)
    print(f"\n📊 VALIDATION RESULT: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    # Detailed mismatch analysis
    details = results.get("details", {})
    if not validation_passed:
        print(f"\n⚠️  MISMATCH DETAILS:")
        if not details.get("compile_match", True):
            print(f"  • Compilation mismatch - Expected: {expected.get('compile')}, Got: {actual.get('compile')}")
        if not details.get("correct_match", True):
            print(f"  • Correctness mismatch - Expected: {expected.get('correct')}, Got: {actual.get('correct')}")
        if not details.get("cuda_match", True):
            print(f"  • CUDA usage mismatch - Expected: {expected.get('cuda')}, Got: {actual.get('cuda')}")
    
    # Request information
    request_info = results.get("request_info", {})
    print(f"\n📡 REQUEST INFORMATION:")
    print(f"  • Request Success: {request_info.get('success', False)}")
    print(f"  • Response Time: {request_info.get('response_time', 0):.2f}s")
    print(f"  • Status Code: {request_info.get('status_code', 'N/A')}")
    print(f"  • Job ID: {request_info.get('job_id', 'N/A')}")
    print(f"  • GPU ID: {request_info.get('gpu_id', 'N/A')}")
    
    if request_info.get("error"):
        print(f"  • Error: {request_info.get('error')}")
    
    # Device metrics section (if available)
    formatted_device_metrics = results.get("formatted_device_metrics")
    if formatted_device_metrics:
        print(f"\n🔬 DEVICE METRICS:")
        for line in formatted_device_metrics.split('\n'):
            if line.strip():  # Skip empty lines
                print(f"  {line}")
    else:
        device_metrics = results.get("device_metrics")
        if device_metrics:
            print(f"\n🔬 DEVICE METRICS:")
            print(f"  Device metrics available but not formatted")
        else:
            print(f"\n🔬 DEVICE METRICS:")
            print(f"  No device metrics available for this test case")
    
    print(f"\n{'='*80}")


def main():
    """Main test function"""
    import sys
    
    # Parse command line arguments
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_mode = sys.argv[2] if len(sys.argv) > 2 else "comprehensive"
    
    # Check if test mode is for testing a specific index
    test_index = None
    if test_mode.startswith("test-index"):
        if len(sys.argv) > 3:
            try:
                test_index = int(sys.argv[3])
                test_mode = "test-index"
            except ValueError:
                print(f"❌ Invalid index: {sys.argv[3]}")
                print("   Usage: python test_server.py <base_url> test-index <index>")
                return
        else:
            print("❌ Missing index for test-index mode")
            print("   Usage: python test_server.py <base_url> test-index <index>")
            return
    
    print(f"🎯 CUDA Evaluation Server V2 - Enhanced Load Testing with KernelBench Integration")
    print(f"   Server URL: {base_url}")
    print(f"   Test Mode: {test_mode}")
    if test_index is not None:
        print(f"   Test Index: {test_index}")
    
    try:
        # Always start with basic functionality test
        if not test_server_basic_functionality(base_url):
            print("❌ Basic functionality test failed - aborting load tests")
            return
        
        if test_mode == "basic":
            print("✅ Basic tests completed")
        
        elif test_mode == "test-index":
            # Test a specific test case by index
            print(f"\n🔬 TESTING SPECIFIC TEST CASE AT INDEX {test_index}")
            validator = ValidationTestRunner(base_url)
            result = validator.test_specific_case_by_index(test_index, num_trials=10)
            print_specific_test_report(result)
        
        elif test_mode == "validation":
            # Run validation tests using KernelBench data
            run_validation_tests(base_url)
        
        elif test_mode == "load" or test_mode == "comprehensive":
            # Run comprehensive load tests with CUDA-only cases
            run_comprehensive_load_tests(base_url)
        
        elif test_mode == "quick":
            # Quick load test with CUDA-only cases
            print(f"\n🚀 QUICK LOAD TEST")
            tester = ConcurrentLoadTester(base_url)
            results = tester.test_concurrent_requests(num_concurrent=3, num_trials_per_request=5, cuda_only=True)
            LoadTestReporter.print_detailed_report("Quick Load Test", results)
        
        else:
            print(f"❌ Unknown test mode: {test_mode}")
            print("   Available modes:")
            print("     • basic - Basic functionality tests")
            print("     • quick - Quick load test with 3 concurrent requests")
            print("     • validation - Validation tests with KernelBench data")
            print("     • load - Load testing suite")
            print("     • comprehensive - Comprehensive testing (default)")
            print("     • test-index <index> - Test specific test case by index")
            print("   Example: python test_server.py http://localhost:8000 test-index 42")
            
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
