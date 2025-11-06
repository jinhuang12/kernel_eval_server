"""
Integration tests for failure handling and edge cases
"""

import pytest
import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.kernels import KernelLibrary
from tests.fixtures.factories import RequestFactory, ResponseValidator


@pytest.mark.integration
class TestCompilationFailures:
    """Tests for compilation failure scenarios"""
    
    def test_syntax_error_handling(self, server_url, test_index):
        """Test handling of syntax errors in kernel code"""
        kernel = KernelLibrary.compilation_error()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 200
        data = response.json()

        results = data["kernel_exec_result"]
        assert results["compiled"] == False
        assert "compilation_error" in results
        assert "syntax" in results["compilation_error"].lower() or "compilation" in results["compilation_error"].lower()
    
    def test_missing_function_handling(self, server_url, test_index):
        """Test handling when kernel_fn is missing"""
        kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch

# No kernel_fn defined
def some_other_function(x):
    return x * 2
"""
        }

        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200
        data = response.json()

        results = data["kernel_exec_result"]
        assert results["compiled"] == False
        assert "compilation_error" in results

    def test_import_error_handling(self, server_url, test_index):
        """Test handling of import errors"""
        kernel = {
            "kernel_type": "torch",
            "source_code": """
import nonexistent_module

def kernel_fn(x):
    return x
"""
        }

        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3
        )

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200
        data = response.json()

        results = data["kernel_exec_result"]
        assert results["compiled"] == False
        assert "compilation_error" in results


@pytest.mark.integration
class TestValidationFailures:
    """Tests for validation failure scenarios"""
    
    def test_incorrect_output_handling(self, server_url, test_index):
        """Test handling when target produces incorrect output"""
        ref_kernel = KernelLibrary.torch_add()
        target_kernel = KernelLibrary.validation_failure()  # Returns subtraction instead
        
        request = RequestFactory.create_compare_request(
            ref_kernel=ref_kernel,
            custom_kernel=target_kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/compare", json=request)

        assert response.status_code == 200
        data = response.json()

        results = data["kernel_exec_result"]
        assert results["compiled"] == True
        assert results["correctness"] == False
        assert "validation_error" in results
    
    def test_shape_mismatch_handling(self, server_url, test_index):
        """Test handling of shape mismatches"""
        ref_kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch

def kernel_fn(x):
    return x.reshape(-1)  # Flatten
"""
        }
        
        target_kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch

def kernel_fn(x):
    return x.reshape(-1, 1)  # Different shape
"""
        }
        
        request = RequestFactory.create_compare_request(
            ref_kernel=ref_kernel,
            custom_kernel=target_kernel,
            num_trials=3
        )
        
        response = requests.post(f"{server_url}/compare", json=request)

        assert response.status_code == 200
        data = response.json()

        results = data["kernel_exec_result"]
        assert results["correctness"] == False
        assert "shape" in str(results.get("validation_error", "")).lower()


@pytest.mark.integration
class TestClientErrors:
    """Tests for HTTP 400 client error scenarios"""
    
    def test_invalid_enum_value(self, server_url):
        """Test invalid kernel_type enum value"""
        request = {
            "index": "test",
            "ref_kernel": {
                "kernel_type": "invalid_type",
                "source_code": "pass"
            },
            "num_trials": 3
        }
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data or "validation_errors" in data
    
    def test_missing_required_field(self, server_url):
        """Test missing required field"""
        request = {
            "index": "test",
            # Missing ref_kernel or target_kernel
            "num_trials": 3
        }
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 400
    
    def test_invalid_num_trials(self, server_url):
        """Test invalid num_trials value"""
        kernel = KernelLibrary.torch_add()
        
        request = RequestFactory.create_evaluate_request(
            index="test",
            kernel=kernel,
            num_trials=0  # Invalid: must be >= 1
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 400
        data = response.json()
        assert "num_trials" in str(data).lower()
    
    def test_invalid_timeout(self, server_url):
        """Test invalid timeout value"""
        kernel = KernelLibrary.torch_add()
        
        request = RequestFactory.create_evaluate_request(
            index="test",
            kernel=kernel,
            timeout=2  # Invalid: must be >= 5
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 400
        data = response.json()
        assert "timeout" in str(data).lower()
    
    def test_triton_without_iocontract(self, server_url):
        """Test Triton kernel without required IOContract"""
        request = {
            "index": "test",
            "ref_kernel": {
                "kernel_type": "triton",
                "source_code": "pass"
                # Missing io_contract
            },
            "num_trials": 3
        }
        
        response = requests.post(f"{server_url}/evaluate", json=request)
        
        assert response.status_code == 400
        data = response.json()
        assert "io_contract" in str(data).lower()


@pytest.mark.integration
@pytest.mark.triton
class TestTritonErrorReporting:
    """Tests for Triton-specific error reporting"""

    def test_triton_constraint_error_reporting(self, server_url):
        """Test that Triton constraint errors show full stack trace"""
        # The Triton kernel that will fail due to M < 16 constraint in tl.dot()
        kernel_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def small_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # This will fail because BLOCK_SIZE < 16
        accumulator += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
"""

        # IOContract with small block sizes that will trigger the error
        io_contract = {
            "args": [
                {
                    "name": "a_ptr",
                    "type": "tensor",
                    "role": "input",
                    "tensor_spec": {
                        "shape": [8, 16],
                        "dtype": "float32",
                        "init": {"kind": "randn", "seed": 42}
                    }
                },
                {
                    "name": "b_ptr",
                    "type": "tensor",
                    "role": "input",
                    "tensor_spec": {
                        "shape": [16, 8],
                        "dtype": "float32",
                        "init": {"kind": "randn", "seed": 43}
                    }
                },
                {
                    "name": "c_ptr",
                    "type": "tensor",
                    "role": "output",
                    "tensor_spec": {
                        "shape": [8, 8],
                        "dtype": "float32"
                    }
                },
                {"name": "M", "type": "int", "value": 8},
                {"name": "N", "type": "int", "value": 8},
                {"name": "K", "type": "int", "value": 16},
                {"name": "stride_am", "type": "int", "value": 16},
                {"name": "stride_ak", "type": "int", "value": 1},
                {"name": "stride_bk", "type": "int", "value": 8},
                {"name": "stride_bn", "type": "int", "value": 1},
                {"name": "stride_cm", "type": "int", "value": 8},
                {"name": "stride_cn", "type": "int", "value": 1},
                {"name": "BLOCK_SIZE_M", "type": "int", "value": 8, "is_meta": True},
                {"name": "BLOCK_SIZE_N", "type": "int", "value": 8, "is_meta": True},
                {"name": "BLOCK_SIZE_K", "type": "int", "value": 8, "is_meta": True}
            ],
            "launch": {
                "grid": {"x": 1, "y": 1, "z": 1},
                "num_warps": 1,
                "num_stages": 1
            }
        }

        # Create the request
        request = {
            "kernel": {
                "source_code": kernel_code,
                "kernel_type": "triton",
                "io": io_contract
            },
            "num_trials": 10,
            "collect_device_metrics": False
        }

        # Send request to server
        response = requests.post(f"{server_url}/evaluate", json=request)
        result = response.json()

        # Should return 200 with validation error containing full trace
        assert response.status_code == 200

        # Check for validation error with stack trace
        if "kernel_exec_result" in result:
            kernel_exec = result["kernel_exec_result"]

            # If validation failed, check error message contains helpful info
            if not kernel_exec.get("correctness", True):
                val_error = kernel_exec.get("validation_error", "")

                # Error message should contain helpful debugging information
                assert len(val_error) > 0, "Validation error should not be empty"

                # Check that error contains relevant keywords
                # (the specific error format may vary, but should be informative)
                assert "error" in val_error.lower() or "failed" in val_error.lower()


@pytest.mark.integration
class TestCrashHandling:
    """Tests for subprocess crash handling"""

    def test_segmentation_fault(self, server_url, test_index):
        """Test handling of segmentation fault in subprocess"""
        kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch
import ctypes

def kernel_fn(x):
    ctypes.string_at(0)  # Causes segfault
    return x
"""
        }
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=3,
            timeout=10
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)

        # Should handle crash gracefully
        assert response.status_code == 200
        data = response.json()

        # Should have partial results
        assert "kernel_exec_result" in data
        assert data["kernel_exec_result"]["compiled"] == False
        assert "compilation_error" in data["kernel_exec_result"]
    
    def test_infinite_loop_timeout(self, server_url, test_index):
        """Test handling of infinite loops via timeout"""
        kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch
import time

def kernel_fn(x):
    while True:
        time.sleep(0.1)
    return x
"""
        }
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=1,
            timeout=5  # Short timeout
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request, timeout=30)
        
        # Should timeout and return partial results
        assert response.status_code in [200, 408, 504]
    
    def test_out_of_memory(self, server_url, test_index):
        """Test handling of out-of-memory errors"""
        kernel = {
            "kernel_type": "torch",
            "source_code": """
import torch

def kernel_fn(x):
    # Try to allocate huge tensor
    huge = torch.zeros(1000000, 1000000, dtype=torch.float32, device='cuda')
    return x
"""
        }
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=1,
            timeout=10
        )
        
        response = requests.post(f"{server_url}/evaluate", json=request)

        # Should handle OOM gracefully
        assert response.status_code == 200
        data = response.json()

        results = data["kernel_exec_result"]
        assert results["compiled"] == False or results.get("profiling_error") is not None
        assert "compilation_error" in results or "profiling_error" in results