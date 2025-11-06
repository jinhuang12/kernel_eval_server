"""
Common utilities and helpers for CUDA Evaluation Server testing
"""

import json
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes"""
    BASIC = "basic"
    QUICK = "quick"
    VALIDATION = "validation"
    LOAD = "load"
    PERFORMANCE = "performance"
    COMPREHENSIVE = "comprehensive"
    TRITON = "triton"
    DEBUG = "debug"


class KernelType(Enum):
    """Supported kernel types"""
    TORCH = "torch"
    TORCH_CUDA = "torch_cuda"
    TRITON = "triton"
    CUDA = "cuda"


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    success: bool
    duration: float
    status_code: int = 0
    error: Optional[str] = None
    job_id: Optional[str] = None
    gpu_id: Optional[int] = None
    # Validation fields
    expected_compile: Optional[bool] = None
    expected_correct: Optional[bool] = None
    actual_compile: Optional[bool] = None
    actual_correct: Optional[bool] = None
    # Performance fields
    ref_runtime_ms: Optional[float] = None
    custom_runtime_ms: Optional[float] = None
    speedup: Optional[float] = None
    # Metrics
    device_metrics: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TritonTestCase:
    """Container for Triton kernel test cases"""
    name: str
    description: str
    ref_kernel: Dict[str, Any]  # PyTorch reference
    custom_kernel: Dict[str, Any]  # Triton implementation
    optimization_level: str = "naive"
    expected_speedup: float = 1.0
    category: str = "general"
    

@dataclass
class TestConfig:
    """Configuration for test execution"""
    base_url: str = "http://localhost:8000"
    num_trials: int = 10
    timeout: int = 120
    batch_size: int = 10
    concurrent_requests: int = 5
    enable_device_metrics: bool = False
    verbose: bool = False
    test_data_path: str = "test_data"
    kernelbench_json: str = "test_data/kernelbench_evaluated_20250808_212140.json"
    test_mode: TestMode = TestMode.COMPREHENSIVE
    kernel_types: List[KernelType] = field(default_factory=lambda: [KernelType.TORCH_CUDA])
    triton_optimization_level: str = "naive"
    triton_test_dir: str = "test_data/triton"
    enable_iocontract: bool = False


class TestCaseProvider:
    """Unified provider for test cases from KernelBench JSON"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.test_cases = []
        self.cuda_cases = []
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test cases from JSON file"""
        if not os.path.exists(self.json_path):
            logger.warning(f"Test data file not found: {self.json_path}")
            return
        
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            for idx, case in enumerate(data):
                if self._validate_case(case):
                    test_case = {
                        "index": idx,
                        "ref_code": case['ref'],
                        "custom_code": case['generated'],
                        "expected_compile": case.get('compile', None),
                        "expected_correct": case.get('correct', None),
                        "expected_cuda": case.get('cuda', None),
                        "ref_runtime_ms": case.get('ref_runtime_ms'),
                        "generated_runtime_ms": case.get('generated_runtime_ms'),
                        "speedup": case.get('speedup')
                    }
                    self.test_cases.append(test_case)
                    
                    if case.get('cuda'):
                        self.cuda_cases.append(test_case)
            
            logger.info(f"Loaded {len(self.test_cases)} test cases ({len(self.cuda_cases)} CUDA)")
            
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
    
    def _validate_case(self, case: Dict) -> bool:
        """Validate that a test case has required fields"""
        required = ['ref', 'generated']
        return all(key in case for key in required)
    
    def get_case(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific test case by index"""
        if 0 <= index < len(self.test_cases):
            return self.test_cases[index]
        return None
    
    def get_random_case(self, cuda_only: bool = False) -> Optional[Dict[str, Any]]:
        """Get a random test case"""
        source = self.cuda_cases if cuda_only else self.test_cases
        return random.choice(source) if source else None
    
    def get_batch(self, size: int, cuda_only: bool = False) -> List[Dict[str, Any]]:
        """Get a batch of test cases"""
        source = self.cuda_cases if cuda_only else self.test_cases
        if not source:
            return []
        sample_size = min(size, len(source))
        return random.sample(source, sample_size)


class ServerClient:
    """HTTP client for server communication"""
    
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def health_check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check server health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.ok else {}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False, {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=5)
            return response.json() if response.ok else {}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def submit_evaluation(self, request_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Submit kernel evaluation request"""
        try:
            response = requests.post(
                f"{self.base_url}/",
                json=request_data,
                timeout=self.timeout
            )
            return response.ok, response.json() if response.text else {"status_code": response.status_code}
        except requests.Timeout:
            return False, {"error": "Request timeout"}
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_gpu_count(self) -> int:
        """Get number of available GPUs"""
        success, health_data = self.health_check()
        if success:
            profiling = health_data.get("profiling_service", {})
            return profiling.get("total_gpus", 0)
        return 0


class ResultAnalyzer:
    """Analyze and extract information from server responses"""
    
    @staticmethod
    def extract_compilation_status(response: Dict[str, Any]) -> Optional[bool]:
        """Extract whether compilation succeeded"""
        # First check kernel_exec_result for explicit compilation status
        kernel_result = response.get('kernel_exec_result', {})
        if isinstance(kernel_result, dict) and 'compiled' in kernel_result:
            return kernel_result['compiled']
        
        # Fallback to status and error message analysis
        status = response.get('status')
        if status == 'success':
            return True  # If overall success, compilation must have worked
        
        error_msg = (response.get('error_message', '') or 
                    response.get('error', '')).lower()
        
        if 'compilation failed' in error_msg:
            return False
        if 'validation failed' in error_msg or 'correctness' in error_msg:
            return True  # Compiled but validation failed
        
        return None
    
    @staticmethod
    def extract_correctness_status(response: Dict[str, Any]) -> Optional[bool]:
        """Extract whether correctness validation passed"""
        # First check kernel_exec_result for explicit correctness status
        kernel_result = response.get('kernel_exec_result', {})
        if isinstance(kernel_result, dict) and 'correctness' in kernel_result:
            return kernel_result['correctness']
        
        # Fallback to status and error message analysis
        status = response.get('status')
        error_msg = (response.get('error_message', '') or 
                    response.get('error', '')).lower()
        
        if 'compilation failed' in error_msg:
            return None  # Can't determine if didn't compile
        if 'validation failed' in error_msg or 'correctness' in error_msg:
            return False
        
        # Only return True if status is success AND no validation errors
        if status == 'success' and 'validation_error' not in response.get('kernel_exec_result', {}):
            return True
        
        return None
    
    @staticmethod
    def extract_performance_metrics(response: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from response"""
        metrics = {}
        
        # Extract reference runtime
        ref_runtime = response.get('ref_runtime', {})
        if isinstance(ref_runtime, dict):
            metrics['ref_runtime_ms'] = ref_runtime.get('mean', 0)
        else:
            metrics['ref_runtime_ms'] = ref_runtime or 0
        
        # Extract custom runtime
        kernel_result = response.get('kernel_exec_result', {})
        metrics['custom_runtime_ms'] = kernel_result.get('runtime', 0)
        
        # Calculate speedup
        if metrics['ref_runtime_ms'] > 0 and metrics['custom_runtime_ms'] > 0:
            metrics['speedup'] = metrics['ref_runtime_ms'] / metrics['custom_runtime_ms']
        else:
            metrics['speedup'] = 0
        
        return metrics
    
    @staticmethod
    def extract_device_metrics(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract device metrics from response"""
        kernel_result = response.get('kernel_exec_result', {})
        metadata = kernel_result.get('metadata', {})
        return metadata.get('device_metrics')


class MetricsCollector:
    """Collect and aggregate test metrics"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start metrics collection"""
        self.start_time = time.time()
        self.results = []
    
    def stop(self):
        """Stop metrics collection"""
        self.end_time = time.time()
    
    def add_result(self, result: TestResult):
        """Add a test result"""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.results:
            return {"error": "No results collected"}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        durations = [r.duration for r in self.results]
        
        summary = {
            "total_tests": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results),
            "total_duration": self.end_time - self.start_time if self.end_time else 0,
        }
        
        if durations:
            summary["response_times"] = {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0
            }
        
        # GPU utilization
        gpu_ids = [r.gpu_id for r in successful if r.gpu_id is not None]
        if gpu_ids:
            summary["gpu_utilization"] = {
                "unique_gpus": len(set(gpu_ids)),
                "usage_distribution": {
                    gpu_id: gpu_ids.count(gpu_id) 
                    for gpu_id in set(gpu_ids)
                }
            }
        
        # Performance metrics
        speedups = [r.speedup for r in successful if r.speedup]
        if speedups:
            summary["performance"] = {
                "mean_speedup": statistics.mean(speedups),
                "median_speedup": statistics.median(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups)
            }
        
        return summary


class TestReporter:
    """Generate test reports"""
    
    @staticmethod
    def print_summary(title: str, metrics: Dict[str, Any], verbose: bool = False):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        
        if "error" in metrics:
            print(f"❌ Error: {metrics['error']}")
            return
        
        # Basic stats
        print(f"\n📈 Summary:")
        print(f"  Total tests: {metrics.get('total_tests', 0)}")
        print(f"  Successful: {metrics.get('successful', 0)}")
        print(f"  Failed: {metrics.get('failed', 0)}")
        print(f"  Success rate: {metrics.get('success_rate', 0)*100:.1f}%")
        print(f"  Duration: {metrics.get('total_duration', 0):.1f}s")
        
        # Response times
        if "response_times" in metrics:
            rt = metrics["response_times"]
            print(f"\n⏱️ Response Times:")
            print(f"  Mean: {rt['mean']:.2f}s")
            print(f"  Median: {rt['median']:.2f}s")
            print(f"  Range: [{rt['min']:.2f}s, {rt['max']:.2f}s]")
        
        # GPU utilization
        if "gpu_utilization" in metrics:
            gpu = metrics["gpu_utilization"]
            print(f"\n🔧 GPU Utilization:")
            print(f"  Unique GPUs: {gpu['unique_gpus']}")
            if verbose and "usage_distribution" in gpu:
                for gpu_id, count in gpu["usage_distribution"].items():
                    print(f"    GPU {gpu_id}: {count} requests")
        
        # Performance
        if "performance" in metrics:
            perf = metrics["performance"]
            print(f"\n🚀 Performance:")
            print(f"  Mean speedup: {perf['mean_speedup']:.2f}x")
            print(f"  Median speedup: {perf['median_speedup']:.2f}x")
    
    @staticmethod
    def save_report(filepath: str, metrics: Dict[str, Any]):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Report saved to {filepath}")


class IOContractTestProvider:
    """Provider for IOContract-based test cases from JSON files"""
    
    def __init__(self, test_dir: str = "test_data/triton"):
        self.test_dir = test_dir
        self.test_cases = []
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test cases from JSON files in test directory"""
        import glob
        json_files = glob.glob(os.path.join(self.test_dir, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Create TritonTestCase from JSON data
                test_name = os.path.basename(json_file).replace('.json', '')
                if 'ref_kernel' in data and 'custom_kernel' in data:
                    # Validate kernel_type fields are strings
                    if 'kernel_type' in data['ref_kernel']:
                        if not isinstance(data['ref_kernel']['kernel_type'], str):
                            logger.warning(f"Converting ref_kernel.kernel_type to string in {test_name}")
                            data['ref_kernel']['kernel_type'] = str(data['ref_kernel']['kernel_type'])
                    
                    if 'kernel_type' in data['custom_kernel']:
                        if not isinstance(data['custom_kernel']['kernel_type'], str):
                            logger.warning(f"Converting custom_kernel.kernel_type to string in {test_name}")
                            data['custom_kernel']['kernel_type'] = str(data['custom_kernel']['kernel_type'])
                    
                    test_case = TritonTestCase(
                        name=test_name,
                        description=f"IOContract test: {test_name}",
                        ref_kernel=data['ref_kernel'],
                        custom_kernel=data['custom_kernel'],
                        category="iocontract"
                    )
                    self.test_cases.append(test_case)
                    logger.info(f"Loaded IOContract test: {test_name}")
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.test_cases)} IOContract test cases")
    
    def get_test_case(self, name: str) -> Optional[TritonTestCase]:
        """Get test case by name"""
        for test_case in self.test_cases:
            if test_case.name == name:
                return test_case
        return None
    
    def get_all_test_cases(self) -> List[TritonTestCase]:
        """Get all test cases"""
        return self.test_cases


def create_evaluation_request(
    test_case: Any,
    num_trials: int = 10,
    timeout: int = 60,
    kernel_type: str = "torch_cuda",
    enable_device_metrics: bool = False
) -> Dict[str, Any]:
    """Create an evaluation request from a test case (supports multiple formats)"""
    
    # Handle TritonTestCase format (ref_kernel/custom_kernel)
    if isinstance(test_case, TritonTestCase) or (isinstance(test_case, dict) and 'ref_kernel' in test_case):
        if isinstance(test_case, TritonTestCase):
            ref_kernel = test_case.ref_kernel.copy()  # Make a copy to avoid modifying original
            custom_kernel = test_case.custom_kernel.copy()
        else:
            ref_kernel = test_case['ref_kernel'].copy()
            custom_kernel = test_case['custom_kernel'].copy()
        
        # Ensure kernel_type fields are strings, not enum objects
        if 'kernel_type' in ref_kernel:
            if hasattr(ref_kernel['kernel_type'], 'value'):
                ref_kernel['kernel_type'] = ref_kernel['kernel_type'].value
            elif not isinstance(ref_kernel['kernel_type'], str):
                ref_kernel['kernel_type'] = str(ref_kernel['kernel_type'])
                
        if 'kernel_type' in custom_kernel:
            if hasattr(custom_kernel['kernel_type'], 'value'):
                custom_kernel['kernel_type'] = custom_kernel['kernel_type'].value
            elif not isinstance(custom_kernel['kernel_type'], str):
                custom_kernel['kernel_type'] = str(custom_kernel['kernel_type'])
            
        request = {
            "ref_kernel": ref_kernel,
            "custom_kernel": custom_kernel,
            "num_trials": num_trials,
            "timeout": timeout
        }
    # Handle traditional format (ref_code/custom_code)
    elif isinstance(test_case, dict) and 'ref_code' in test_case:
        request = {
            "ref_code": test_case["ref_code"],
            "custom_code": test_case["custom_code"],
            "num_trials": num_trials,
            "timeout": timeout,
            "kernel_type": kernel_type
        }
    else:
        raise ValueError(f"Unknown test case format: {type(test_case)}")
    
    if enable_device_metrics:
        request["enable_device_metrics"] = True
    
    # Debug log the request structure (only kernel types, not full code)
    if logger.isEnabledFor(logging.DEBUG):
        debug_request = {k: v for k, v in request.items() if k not in ['ref_code', 'custom_code']}
        if 'ref_kernel' in request:
            debug_request['ref_kernel_type'] = request['ref_kernel'].get('kernel_type', 'unknown')
        if 'custom_kernel' in request:
            debug_request['custom_kernel_type'] = request['custom_kernel'].get('kernel_type', 'unknown')
        logger.debug(f"Created evaluation request: {debug_request}")
    
    return request


def run_with_timeout(func, timeout: int, *args, **kwargs):
    """Run a function with timeout (simplified for macOS compatibility)"""
    import threading
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return None, "Timeout"
    if exception[0]:
        return None, str(exception[0])
    return result[0], None