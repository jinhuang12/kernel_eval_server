"""
Integration tests for different kernel types
"""

import pytest
import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.kernels import KernelLibrary
from tests.fixtures.factories import RequestFactory, ResponseValidator
from tests.fixtures.test_data_loader import get_loader
from shared.models import KernelType


@pytest.mark.integration
class TestTorchKernels:
    """Integration tests for TORCH kernels"""
    
    @pytest.mark.parametrize("kernel_name", [
        "torch_add",
        "torch_matmul",
        "torch_gelu"
    ])
    def test_torch_kernel_execution(self, server_url, kernel_name):
        """Test executing various Torch kernels"""
        kernel = KernelLibrary.get_kernel(kernel_name)
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=5
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert ResponseValidator.validate_compilation_success(data)
        assert ResponseValidator.validate_performance_metrics(data)
    
    def test_torch_kernel_comparison(self, server_url):
        """Test comparing two Torch kernels"""
        ref_kernel = KernelLibrary.torch_add()
        custom_kernel = KernelLibrary.torch_add()  # Same operation
        
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


@pytest.mark.integration
@pytest.mark.triton
class TestTritonKernels:
    """Integration tests for TRITON kernels"""
    
    def test_triton_kernel_with_iocontract(self, server_url):
        """Test Triton kernel with IOContract"""
        kernel = KernelLibrary.triton_add()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=5
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert ResponseValidator.validate_compilation_success(data)
        assert ResponseValidator.validate_performance_metrics(data)
    
    def test_triton_without_iocontract(self, server_url):
        """Test that Triton kernel without IOContract fails properly"""
        kernel = {
            "kernel_type": "triton",
            "kernel_name": "invalid_triton",
            "kernel_code": "pass"
        }
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 400
        data = response.json()
        assert "io_contract" in str(data).lower()
    
    def test_triton_matmul_performance(self, server_url):
        """Test optimized Triton matmul performance"""
        kernel = KernelLibrary.triton_matmul()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=10
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert ResponseValidator.validate_compilation_success(data)
        
        # Check performance metrics are reasonable
        ker = data["kernel_exec_result"]
        if "runtime_stats" in ker:
            perf = ker["runtime_stats"]
            assert perf["mean"] > 0
            assert perf["std"] >= 0
            assert perf["min"] <= perf["mean"] <= perf["max"]
    
    def test_triton_vs_torch_comparison(self, server_url):
        """Test comparing Triton vs Torch implementation"""
        ref_kernel = KernelLibrary.torch_add()
        custom_kernel = KernelLibrary.triton_add()
        
        request = RequestFactory.create_compare_request(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            num_trials=5
        )
        
        response = requests.post(f"{server_url}/compare", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should compile both
        assert data["kernel_exec_result"]["compiled"]
        
        # Correctness check should pass
        assert data["kernel_exec_result"]["correctness"]
        
        # Should have performance metrics
        assert "runtime" in data["kernel_exec_result"]
        assert "ref_runtime" in data


@pytest.mark.integration
@pytest.mark.cuda
class TestTorchCudaKernels:
    """Integration tests for TORCH_CUDA kernels"""
    
    def test_torch_cuda_kernel(self, server_url):
        """Test PyTorch with embedded CUDA kernel"""
        kernel = KernelLibrary.torch_cuda_add()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=5,
            timeout=60  # May take longer to compile
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # May or may not compile successfully depending on environment
        if data["kernel_exec_result"]["compiled"]:
            assert ResponseValidator.validate_performance_metrics(data)
        else:
            assert "compilation_error" in data["kernel_exec_result"]


@pytest.mark.integration
@pytest.mark.cuda
class TestCudaKernelsFromTestData:
    """Test CUDA kernels from actual test data JSON files"""
    
    def test_cuda_kernel_evaluation(self, server_url, cuda_test_case):
        """Test evaluating CUDA kernels from test data"""
        kernel = cuda_test_case.get("custom_kernel") or cuda_test_case.get("ref_kernel")
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=cuda_test_case.get("num_trials", 5),
            timeout=cuda_test_case.get("timeout", 30)
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()
        assert ResponseValidator.validate_compilation_success(data)


@pytest.mark.integration
class TestMixedKernelTypes:
    """Test mixing different kernel types"""
    
    def test_compare_across_types(self, server_url):
        """Test comparing kernels of different types"""
        test_cases = [
            ("torch_add", "torch_add"),  # Same type
            ("torch_add", "triton_add"),  # Different types
        ]

        for ref_name, target_name in test_cases:
            ref_kernel = KernelLibrary.get_kernel(ref_name)
            target_kernel = KernelLibrary.get_kernel(target_name)
            
            if not ref_kernel or not target_kernel:
                pytest.skip(f"Kernel not available: {ref_name} or {target_name}")
            
            request = RequestFactory.create_compare_request(
                ref_kernel=ref_kernel,
                custom_kernel=target_kernel,
                num_trials=3
            )
            
            response = requests.post(f"{server_url}/compare", json=request)
            
            assert response.status_code == 200
            data = response.json()
            
            if data["kernel_exec_result"]["compiled"]:
                # If both compiled, check correctness
                assert "correctness" in data["kernel_exec_result"]
    
    def test_kernel_type_performance_comparison(self, server_url):
        """Compare performance across kernel types for same operation"""
        kernels_to_test = [
            ("torch", "torch_add"),
            ("triton", "triton_add"),
        ]

        performances = {}

        for kernel_type, kernel_name in kernels_to_test:
            kernel = KernelLibrary.get_kernel(kernel_name)
            if not kernel:
                continue
            
            request = RequestFactory.create_evaluate_request(
                kernel=kernel,
                num_trials=10
            )
            
            response = requests.post(f"{server_url}/evaluate", json=request)
            
            if response.status_code == 200:
                data = response.json()
                if data["kernel_exec_result"]["compiled"]:
                    perf = data["kernel_exec_result"]["runtime_stats"]
                    performances[kernel_type] = perf["mean"]
        
        # Just verify we got some performance data
        assert len(performances) > 0
        
        # Log relative performance if we have multiple
        if len(performances) > 1:
            baseline = performances.get("torch", 1.0)
            for kernel_type, time in performances.items():
                speedup = baseline / time if time > 0 else 0
                print(f"{kernel_type}: {time:.6f}s (speedup: {speedup:.2f}x)")


@pytest.mark.integration
class TestKernelEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_kernel_with_no_inputs(self, server_url):
        """Test kernel that takes no inputs"""
        kernel = {
            "kernel_type": "torch",
            "kernel_name": "constant_kernel",
            "kernel_code": """
import torch

def kernel_fn():
    return torch.ones(10, 10)
"""
        }
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_kernel_with_many_outputs(self, server_url):
        """Test kernel returning multiple values"""
        kernel = {
            "kernel_type": "torch",
            "kernel_name": "multi_output",
            "kernel_code": """
import torch

def kernel_fn(x):
    return x * 2, x * 3, x.sum()
"""
        }
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200