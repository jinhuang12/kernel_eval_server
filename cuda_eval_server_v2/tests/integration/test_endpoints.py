"""
Integration tests for API endpoints
"""

import pytest
import requests
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.kernels import KernelLibrary
from tests.fixtures.factories import RequestFactory, ResponseValidator, TestResult
from tests.fixtures.validators import TestTimer
from tests.fixtures.test_data_loader import get_loader


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_check(self, server_url):
        """Test basic health check"""
        response = requests.get(f"{server_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_with_timeout(self, server_url):
        """Test health check with custom timeout"""
        response = requests.get(f"{server_url}/health", timeout=5)
        
        assert response.status_code == 200


@pytest.mark.integration
class TestEvaluateEndpoint:
    """Tests for /evaluate endpoint"""
    
    def test_evaluate_single_kernel(self, server_url):
        """Test evaluating a single kernel"""
        kernel = KernelLibrary.torch_add()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=5,
            timeout=30
        )
        
        with TestTimer() as timer:
            response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert ResponseValidator.validate_success_response(data)
        assert ResponseValidator.validate_compilation_success(data)
        assert ResponseValidator.validate_performance_metrics(data)
        
        # Check timing
        assert timer.duration() < 35  # Should complete within timeout + buffer
    
    def test_evaluate_with_metrics(self, server_url):
        """Test evaluation with metrics collection"""
        kernel = KernelLibrary.torch_add()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check if device metrics present (optional based on server config)
        if ResponseValidator.validate_device_metrics(data):
            assert isinstance(data["kernel_exec_result"]["metadata"]["device_metrics"], dict)
    
    @pytest.mark.parametrize("kernel_type,kernel_name", [
        ("torch", "torch_add"),
        ("torch", "torch_gelu"),
        pytest.param("triton", "triton_add", marks=pytest.mark.triton),
    ])
    def test_evaluate_different_kernels(self, server_url, kernel_type, kernel_name):
        """Test evaluating different kernel types"""
        kernel = KernelLibrary.get_kernel(kernel_name)
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert ResponseValidator.validate_compilation_success(data)
    
    def test_evaluate_with_real_test_data(self, server_url, test_data_loader):
        """Test evaluating kernels from actual JSON test data"""
        # Get a test case
        test_cases = test_data_loader.get_cuda_test_cases()
        if not test_cases:
            pytest.skip("No CUDA test cases available")
        
        name, test_case = test_cases[0]  # Use first test case
        kernel = test_case.get("custom_kernel") or test_case.get("ref_kernel")
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert ResponseValidator.validate_compilation_success(data)


@pytest.mark.integration
class TestRuntimeStatistics:
    """Tests for runtime statistics in responses"""

    def test_percentiles_in_runtime_stats(self, server_url):
        """Test that runtime stats include all expected fields including percentiles"""
        kernel = KernelLibrary.torch_add()

        request = RequestFactory.create_compare_request(
            ref_kernel=kernel,
            custom_kernel=kernel,
            num_trials=10
        )

        response = requests.post(f"{server_url}/compare", json=request)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        result = response.json()

        # Check if we have runtime stats
        kernel_exec = result.get("kernel_exec_result", {})
        runtime_stats = kernel_exec.get("runtime_stats", {})
        ref_runtime = result.get("ref_runtime", {})

        # Expected fields in runtime statistics
        expected_fields = ["mean", "std", "min", "max", "median", "percentile_95", "percentile_99"]

        # Verify custom kernel runtime_stats
        if runtime_stats:
            for field in expected_fields:
                assert field in runtime_stats, f"Missing {field} in runtime_stats"
                assert runtime_stats[field] is not None, f"{field} is None in runtime_stats"

        # Verify reference kernel runtime
        if ref_runtime:
            for field in expected_fields:
                assert field in ref_runtime, f"Missing {field} in ref_runtime"
                assert ref_runtime[field] is not None, f"{field} is None in ref_runtime"

    def test_runtime_stats_consistency(self, server_url):
        """Test that runtime statistics are mathematically consistent"""
        kernel = KernelLibrary.torch_add()

        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=10
        )

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200
        result = response.json()

        kernel_exec = result.get("kernel_exec_result", {})
        runtime_stats = kernel_exec.get("runtime_stats", {})

        if runtime_stats:
            # Verify consistency: min <= mean <= max
            assert runtime_stats["min"] <= runtime_stats["mean"]
            assert runtime_stats["mean"] <= runtime_stats["max"]

            # Verify median is between min and max
            assert runtime_stats["min"] <= runtime_stats["median"] <= runtime_stats["max"]

            # Verify percentiles are ordered: median <= p95 <= p99
            assert runtime_stats["median"] <= runtime_stats["percentile_95"]
            assert runtime_stats["percentile_95"] <= runtime_stats["percentile_99"]

            # Verify p99 <= max
            assert runtime_stats["percentile_99"] <= runtime_stats["max"]

            # Verify std is non-negative
            assert runtime_stats["std"] >= 0


@pytest.mark.integration
class TestCompareEndpoint:
    """Tests for /compare endpoint"""

    def test_compare_identical_kernels(self, server_url):
        """Test comparing identical kernels"""
        kernel = KernelLibrary.torch_add()

        request = RequestFactory.create_compare_request(
            ref_kernel=kernel,
            custom_kernel=kernel,
            num_trials=5
        )

        response = requests.post(f"{server_url}/compare", json=request)

        assert response.status_code == 200
        data = response.json()

        assert ResponseValidator.validate_success_response(data)
        assert ResponseValidator.validate_correctness(data)

        # Performance should be similar
        perf = ResponseValidator.extract_performance(data)
        if "speedup" in perf:
            assert 0.8 < perf["speedup"] < 1.2  # Within 20%
    
    def test_compare_different_implementations(self, server_url):
        """Test comparing different implementations of same operation"""
        # Load actual test case from JSON
        loader = get_loader()
        test_case = loader.get_test_case("vector_add")
        
        if test_case:
            ref_kernel = test_case.get("ref_kernel")
            custom_kernel = test_case.get("custom_kernel")
        else:
            # Fallback to hardcoded
            ref_kernel = KernelLibrary.torch_add()
            custom_kernel = ref_kernel.copy()
            custom_kernel["kernel_type"] = "torch_cuda"
        
        request = RequestFactory.create_compare_request(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            num_trials=5
        )
        
        response = requests.post(f"{server_url}/compare", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert ResponseValidator.validate_correctness(data)
        assert ResponseValidator.validate_speedup(data)
    
    def test_compare_with_validation_failure(self, server_url):
        """Test comparison when validation fails"""
        ref_kernel = KernelLibrary.torch_add()
        # Create a kernel that returns wrong results
        custom_kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch

def get_inputs():
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    return [x, y]

def get_init_inputs():
    return []

def kernel_fn(x, y):
    return torch.sub(x, y)  # Wrong operation - should be add
""",
            "io": None,
            "metadata": {"function_name": "kernel_fn"}
        }
        
        request = RequestFactory.create_compare_request(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/compare", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        ker = data["kernel_exec_result"]
        assert ker["compiled"] == True
        assert ker["correctness"] == False
        assert "validation_error" in ker


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling across endpoints"""
    
    def test_invalid_request_format(self, server_url):
        """Test handling of invalid request format"""
        invalid_requests = RequestFactory.create_invalid_request()
        
        for request in invalid_requests:
            response = requests.post(f"{server_url}/evaluate", json=request)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data or "validation_errors" in data
    
    def test_compilation_error(self, server_url):
        """Test handling of compilation errors"""
        kernel = KernelLibrary.compilation_error()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200  # Still returns 200
        data = response.json()
        
        ker = data["kernel_exec_result"]
        assert ker["compiled"] == False
        assert "compilation_error" in ker
        print(ker["compilation_error"])
    
    def test_timeout_handling(self, server_url):
        """Test request timeout handling"""
        kernel = KernelLibrary.torch_add()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=100,  # Many trials but not too many
            timeout=10  # Short timeout but reasonable
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        # Should either complete or timeout gracefully
        assert response.status_code in [200, 408, 504]
    
    def test_concurrent_requests(self, server_url):
        """Test handling concurrent requests"""
        kernel = KernelLibrary.torch_add()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        # Send multiple requests concurrently
        import concurrent.futures
        
        def send_request():
            return requests.post(f"{server_url}/evaluate", json=request)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request) for _ in range(3)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200