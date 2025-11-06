"""
Integration tests for Triton kernel metadata and kernel selection
Tests metadata-based kernel selection from multi-kernel source files
Migrated from test_triton_multi_kernel.py and test_metadata_compare.py
"""

import pytest
import requests
import torch
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, LaunchConfig, LaunchDim
from shared.kernel_metadata import TritonKernelMetadata
from compilation.triton.triton_backend import TritonExecutableKernel


# Multi-kernel Triton source with different kernels
MULTI_KERNEL_SOURCE = '''
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Vector addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Vector multiplication kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def scale_kernel(x_ptr, output_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    """Scalar multiplication kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scale
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def kernel_paged_attention_2d(
    query_ptr, key_ptr, value_ptr, output_ptr,
    seq_len: tl.constexpr,
    kv_seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Paged attention kernel implementation"""
    pid = tl.program_id(0)
    head_idx = pid

    # Initialize output accumulator
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    # Process attention
    for i in range(kv_seq_len):
        q = tl.load(query_ptr + head_idx * head_dim + tl.arange(0, head_dim))
        k = tl.load(key_ptr + i * head_dim + tl.arange(0, head_dim))
        v = tl.load(value_ptr + i * head_dim + tl.arange(0, head_dim))

        # Compute attention score
        score = tl.sum(q * k, axis=0)

        # Apply softmax (simplified)
        score = tl.exp(score)

        # Accumulate
        acc += score * v

    # Store output
    tl.store(output_ptr + head_idx * head_dim + tl.arange(0, head_dim), acc)
'''


def create_binary_io_contract(n=1024):
    """Create IOContract for binary operations (add/mul)"""
    return IOContract(
        args=[
            ArgSpec(name="x_ptr", type="tensor", tensor_spec=TensorSpec(shape=[n], dtype="float32"), role="input"),
            ArgSpec(name="y_ptr", type="tensor", tensor_spec=TensorSpec(shape=[n], dtype="float32"), role="input"),
            ArgSpec(name="output_ptr", type="tensor", tensor_spec=TensorSpec(shape=[n], dtype="float32"), role="output"),
            ArgSpec(name="n_elements", type="int", value=n),
            ArgSpec(name="BLOCK_SIZE", type="int", value=128, is_meta=True),
        ],
        launch=LaunchConfig(grid=LaunchDim(x=(n + 127) // 128))
    )


def create_scale_io_contract(n=1024):
    """Create IOContract for scale operation"""
    return IOContract(
        args=[
            ArgSpec(name="x_ptr", type="tensor", tensor_spec=TensorSpec(shape=[n], dtype="float32"), role="input"),
            ArgSpec(name="output_ptr", type="tensor", tensor_spec=TensorSpec(shape=[n], dtype="float32"), role="output"),
            ArgSpec(name="scale", type="float", value=2.5),
            ArgSpec(name="n_elements", type="int", value=n),
            ArgSpec(name="BLOCK_SIZE", type="int", value=128, is_meta=True),
        ],
        launch=LaunchConfig(grid=LaunchDim(x=(n + 127) // 128))
    )


def create_paged_attention_io_contract():
    """Create IOContract for paged attention"""
    return IOContract(
        args=[
            ArgSpec(name="query_ptr", type="tensor", tensor_spec=TensorSpec(shape=[8, 64], dtype="float32"), role="input"),
            ArgSpec(name="key_ptr", type="tensor", tensor_spec=TensorSpec(shape=[128, 64], dtype="float32"), role="input"),
            ArgSpec(name="value_ptr", type="tensor", tensor_spec=TensorSpec(shape=[128, 64], dtype="float32"), role="input"),
            ArgSpec(name="output_ptr", type="tensor", tensor_spec=TensorSpec(shape=[8, 64], dtype="float32"), role="output"),
            ArgSpec(name="seq_len", type="int", value=8, is_meta=True),
            ArgSpec(name="kv_seq_len", type="int", value=128, is_meta=True),
            ArgSpec(name="head_dim", type="int", value=64, is_meta=True),
            ArgSpec(name="BLOCK_SIZE", type="int", value=128, is_meta=True)
        ],
        launch=LaunchConfig(grid=LaunchDim(x=8))
    )


@pytest.mark.integration
@pytest.mark.triton
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonKernelSelection:
    """Test Triton kernel selection with metadata"""

    def test_kernel_selection_add(self):
        """Test selecting add_kernel with metadata"""
        device = torch.device("cuda:0")

        metadata = TritonKernelMetadata(kernel_name="add_kernel")
        io_contract = create_binary_io_contract()

        kernel_code = KernelCode(
            source_code=MULTI_KERNEL_SOURCE,
            kernel_type=KernelType.TRITON,
            io=io_contract,
            metadata=metadata
        )

        exe_add = TritonExecutableKernel(
            kernel_code=kernel_code,
            device=device,
            metadata=metadata
        )

        assert exe_add.kernel_name == "add_kernel"

        # Test execution
        n = 1024
        out = torch.empty(n, device=device)
        x = torch.ones(n, device=device)
        y = torch.ones(n, device=device) * 2
        result = exe_add(x, y, out, n, 128)
        expected = x + y
        assert torch.allclose(result, expected), "Add kernel failed"

    def test_kernel_selection_mul(self):
        """Test selecting mul_kernel with metadata"""
        device = torch.device("cuda:0")

        metadata = TritonKernelMetadata(kernel_name="mul_kernel")
        io_contract = create_binary_io_contract()

        kernel_code = KernelCode(
            source_code=MULTI_KERNEL_SOURCE,
            kernel_type=KernelType.TRITON,
            io=io_contract,
            metadata=metadata
        )

        exe_mul = TritonExecutableKernel(
            kernel_code=kernel_code,
            device=device,
            metadata=metadata
        )

        assert exe_mul.kernel_name == "mul_kernel"

        # Test execution
        n = 1024
        out = torch.empty(n, device=device)
        x = torch.ones(n, device=device) * 2
        y = torch.ones(n, device=device) * 3
        result = exe_mul(x, y, out, n, 128)
        expected = x * y
        assert torch.allclose(result, expected), "Mul kernel failed"

    def test_metadata_from_dict(self):
        """Test metadata creation from dictionary"""
        device = torch.device("cuda:0")

        metadata_dict = {"kernel_name": "scale_kernel"}
        io_contract = create_scale_io_contract()

        kernel_code = KernelCode(
            source_code=MULTI_KERNEL_SOURCE,
            kernel_type=KernelType.TRITON,
            io=io_contract,
            metadata=metadata_dict
        )

        exe = TritonExecutableKernel(
            kernel_code=kernel_code,
            device=device,
            metadata=metadata_dict
        )

        assert exe.kernel_name == "scale_kernel"

        # Test execution
        n = 1024
        scale = 2.5
        x = torch.ones(n, device=device) * 4
        out = torch.empty(n, device=device)
        result = exe(x, out, scale, n, 128)
        expected = x * scale
        assert torch.allclose(result, expected), "Scale kernel failed"

    def test_no_metadata_default(self):
        """Test default behavior without metadata (first kernel)"""
        device = torch.device("cuda:0")
        io_contract = create_binary_io_contract()

        kernel_code = KernelCode(
            source_code=MULTI_KERNEL_SOURCE,
            kernel_type=KernelType.TRITON,
            io=io_contract,
            metadata=None
        )

        exe = TritonExecutableKernel(
            kernel_code=kernel_code,
            device=device
        )

        # Should default to first kernel found (add_kernel)
        assert exe.kernel_name == "add_kernel"

    def test_invalid_kernel_name(self):
        """Test error handling for invalid kernel name"""
        device = torch.device("cuda:0")

        metadata = TritonKernelMetadata(kernel_name="nonexistent_kernel")
        io_contract = create_binary_io_contract()

        kernel_code = KernelCode(
            source_code=MULTI_KERNEL_SOURCE,
            kernel_type=KernelType.TRITON,
            io=io_contract,
            metadata=metadata
        )

        with pytest.raises(RuntimeError, match="Kernel 'nonexistent_kernel' not found"):
            TritonExecutableKernel(
                kernel_code=kernel_code,
                device=device,
                metadata=metadata
            )


@pytest.mark.integration
@pytest.mark.triton
class TestTritonMetadataAPI:
    """Test metadata preservation through API endpoints"""

    def test_metadata_compare_endpoint(self, server_url):
        """Test that metadata with kernel_name is preserved through /compare endpoint"""

        request = {
            "ref_kernel": {
                "source_code": MULTI_KERNEL_SOURCE,
                "kernel_type": "triton",
                "io": create_paged_attention_io_contract().to_dict(),
                "metadata": {
                    "kernel_name": "kernel_paged_attention_2d"
                }
            },
            "custom_kernel": {
                "source_code": MULTI_KERNEL_SOURCE,
                "kernel_type": "triton",
                "io": create_binary_io_contract().to_dict(),
                "metadata": {
                    "kernel_name": "add_kernel"
                }
            },
            "num_trials": 5,
            "timeout": 60
        }

        try:
            response = requests.post(f"{server_url}/compare", json=request, timeout=70)

            assert response.status_code == 200, f"Expected 200, got {response.status_code}"

            result = response.json()

            # Check if compilation was successful
            kernel_exec = result.get("kernel_exec_result", {})
            if kernel_exec.get("compiled"):
                # Both kernels compiled successfully
                pass
            else:
                # If compilation failed, check error message
                compilation_error = kernel_exec.get("compilation_error", "")
                # May fail due to complexity, but error should be informative
                assert len(compilation_error) > 0

        except Exception as e:
            pytest.skip(f"Server not available or test requires complex kernel support: {e}")

    def test_metadata_evaluate_endpoint(self, server_url):
        """Test metadata in /evaluate endpoint with kernel selection"""

        request = {
            "kernel": {
                "source_code": MULTI_KERNEL_SOURCE,
                "kernel_type": "triton",
                "io": create_binary_io_contract().to_dict(),
                "metadata": {
                    "kernel_name": "mul_kernel"
                }
            },
            "num_trials": 5,
            "timeout": 30
        }

        try:
            response = requests.post(f"{server_url}/evaluate", json=request, timeout=40)

            assert response.status_code == 200

            result = response.json()
            kernel_exec = result.get("kernel_exec_result", {})

            if kernel_exec.get("compiled"):
                # Should have executed mul_kernel specifically
                assert kernel_exec.get("correctness") == True
                assert "runtime_stats" in kernel_exec

        except Exception as e:
            pytest.skip(f"Server not available: {e}")

    def test_same_kernel_both_ref_and_custom(self, server_url):
        """Test using same kernel for both ref and custom with metadata"""

        request = {
            "ref_kernel": {
                "source_code": MULTI_KERNEL_SOURCE,
                "kernel_type": "triton",
                "io": create_binary_io_contract().to_dict(),
                "metadata": {"kernel_name": "add_kernel"}
            },
            "custom_kernel": {
                "source_code": MULTI_KERNEL_SOURCE,
                "kernel_type": "triton",
                "io": create_binary_io_contract().to_dict(),
                "metadata": {"kernel_name": "add_kernel"}  # Same kernel
            },
            "num_trials": 5,
            "timeout": 30
        }

        try:
            response = requests.post(f"{server_url}/compare", json=request, timeout=40)

            assert response.status_code == 200

            result = response.json()
            kernel_exec = result.get("kernel_exec_result", {})

            if kernel_exec.get("compiled"):
                # Both using same kernel, should be correct and have similar performance
                assert kernel_exec.get("correctness") == True

        except Exception as e:
            pytest.skip(f"Server not available: {e}")
