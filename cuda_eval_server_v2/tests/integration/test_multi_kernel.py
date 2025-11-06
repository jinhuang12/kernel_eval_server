"""
Integration tests for multi-kernel support
Tests kernels that combine multiple kernel types (Triton + PyTorch + CUDA)
Migrated from root test_multi_kernel.py
"""

import pytest
import requests
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.kernels import KernelLibrary
from tests.fixtures.factories import RequestFactory, ResponseValidator


@pytest.mark.integration
@pytest.mark.triton
class TestMultiKernelEvaluation:
    """Integration tests for multi-kernel evaluation endpoint"""

    def test_simple_multi_kernel_sequence(self, server_url):
        """Test simple multi-kernel combining Triton and PyTorch"""
        # Simple multi-kernel example combining Triton and PyTorch
        source_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def run(x, y):
    '''Entry point: add tensors using Triton, then multiply by 2 using PyTorch'''
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # PyTorch operation
    output = output * 2

    return output
"""

        request = {
            "kernel": {
                "source_code": source_code,
                "kernel_type": "multi_kernel",
                "metadata": {
                    "entry_point": "run",
                    "description": "Add with Triton, multiply with PyTorch"
                },
                "io": {
                    "args": [
                        {
                            "name": "x",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [1024],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 42}
                            },
                            "role": "input"
                        },
                        {
                            "name": "y",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [1024],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 43}
                            },
                            "role": "input"
                        }
                    ]
                }
            },
            "num_trials": 10
        }

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        kernel_result = result.get("kernel_exec_result", {})

        assert kernel_result.get("compiled") == True, "Kernel failed to compile"
        assert kernel_result.get("correctness") == True, "Kernel validation failed"

        runtime_stats = kernel_result.get("runtime_stats", {})
        assert "mean" in runtime_stats, "Missing runtime statistics"

    def test_multi_kernel_error_handling(self, server_url):
        """Test error handling with missing entry point"""
        # Source without the entry point function
        bad_source = """
import torch

def wrong_name(x, y):
    return x + y
"""

        request = {
            "kernel": {
                "source_code": bad_source,
                "kernel_type": "multi_kernel",
                "metadata": {
                    "entry_point": "run",  # This doesn't exist
                    "description": "Test error handling"
                },
                "io": {
                    "args": [
                        {
                            "name": "x",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [1024],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 42}
                            },
                            "role": "input"
                        }
                    ]
                }
            },
            "num_trials": 10
        }

        response = requests.post(f"{server_url}/evaluate", json=request)
        result = response.json()

        # Should return 200 with compilation error
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        kernel_result = result.get("kernel_exec_result", {})
        assert kernel_result.get("compiled") == False, "Should fail compilation"

        error = kernel_result.get("compilation_error", "")
        assert "Entry point 'run' not found" in error, f"Expected entry point error, got: {error}"

    def test_complex_multi_kernel_sequence(self, server_url):
        """Test complex sequence with multiple CUDA and multiple Triton kernels"""
        # Complex source with 2 Triton kernels + 2 CUDA kernels
        complex_source = """
import torch
import triton
import triton.language as tl
import cupy as cp

# Triton Kernel 1: Add two tensors
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Triton Kernel 2: Square the tensor
@triton.jit
def square_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = x * x
    tl.store(output_ptr + offsets, output, mask=mask)

# CUDA Kernel 1: Scale by 2
scale_kernel = cp.RawKernel(r'''
extern "C" __global__
void scale_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}
''', 'scale_kernel')

# CUDA Kernel 2: Add constant
add_constant_kernel = cp.RawKernel(r'''
extern "C" __global__
void add_constant_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + 1.0f;
    }
}
''', 'add_constant_kernel')

def run(x, y):
    '''
    Entry point: Execute sequence of 4 kernels
    1. Triton: Add x + y
    2. CUDA: Scale by 2
    3. Triton: Square
    4. CUDA: Add 1
    '''
    n_elements = x.numel()

    # Step 1: Triton add kernel (x + y)
    temp1 = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, temp1, n_elements, BLOCK_SIZE=1024)

    # Step 2: CUDA scale kernel (temp1 * 2)
    temp2 = torch.empty_like(temp1)
    threads_per_block = 256
    blocks = (n_elements + threads_per_block - 1) // threads_per_block
    scale_kernel(
        (blocks,), (threads_per_block,),
        (cp.asarray(temp1), cp.asarray(temp2), n_elements)
    )

    # Step 3: Triton square kernel (temp2 * temp2)
    temp3 = torch.empty_like(temp2)
    square_kernel[grid](temp2, temp3, n_elements, BLOCK_SIZE=1024)

    # Step 4: CUDA add constant kernel (temp3 + 1)
    output = torch.empty_like(temp3)
    add_constant_kernel(
        (blocks,), (threads_per_block,),
        (cp.asarray(temp3), cp.asarray(output), n_elements)
    )

    return output
"""

        request = {
            "kernel": {
                "source_code": complex_source,
                "kernel_type": "multi_kernel",
                "metadata": {
                    "entry_point": "run",
                    "description": "Complex sequence: 2 Triton + 2 CUDA kernels"
                },
                "io": {
                    "args": [
                        {
                            "name": "x",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [2048],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 42}
                            },
                            "role": "input"
                        },
                        {
                            "name": "y",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [2048],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 43}
                            },
                            "role": "input"
                        }
                    ]
                }
            },
            "num_trials": 10
        }

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        kernel_result = result.get("kernel_exec_result", {})

        assert kernel_result.get("compiled") == True, "Kernel failed to compile"
        assert kernel_result.get("correctness") == True, "Kernel validation failed"

        runtime_stats = kernel_result.get("runtime_stats", {})
        assert "mean" in runtime_stats, "Missing runtime statistics"

    @pytest.mark.parametrize("num_trials", [5, 10, 20])
    def test_multi_kernel_performance(self, server_url, num_trials):
        """Test multi-kernel performance with varying trial counts"""
        source_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def run(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
"""

        request = {
            "kernel": {
                "source_code": source_code,
                "kernel_type": "multi_kernel",
                "metadata": {
                    "entry_point": "run"
                },
                "io": {
                    "args": [
                        {
                            "name": "x",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [1024],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 42}
                            },
                            "role": "input"
                        },
                        {
                            "name": "y",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [1024],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 43}
                            },
                            "role": "input"
                        }
                    ]
                }
            },
            "num_trials": num_trials
        }

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200
        result = response.json()
        kernel_result = result.get("kernel_exec_result", {})

        assert kernel_result.get("compiled") == True
        runtime_stats = kernel_result.get("runtime_stats", {})

        # Verify we have statistical metrics
        assert runtime_stats.get("mean", 0) > 0
        assert runtime_stats.get("std", -1) >= 0
        assert runtime_stats.get("min", 0) <= runtime_stats.get("mean", float("inf"))
        assert runtime_stats.get("max", 0) >= runtime_stats.get("mean", 0)

    @pytest.mark.parametrize("seq_len,workload_name", [(32, "decode"), (512, "prefill")])
    def test_llama3_decoder_layer_production(self, server_url, seq_len, workload_name):
        """
        Production-realistic Llama3-8B decoder layer test

        Multi-kernel components:
        - Triton: RMSNorm (fused layer normalization used in Llama3)
        - CUDA: SiLU activation in SwiGLU MLP
        - PyTorch: QKV projections, attention, linear layers, residuals

        Architecture:
        - hidden_size = 4096
        - num_attention_heads = 32
        - num_kv_heads = 8 (Grouped Query Attention)
        - head_dim = 128
        - intermediate_size = 11008 (SwiGLU MLP)

        Workloads:
        - decode (seq_len=32): Simulates token generation workload
        - prefill (seq_len=512): Simulates prompt processing workload
        """
        batch_size = 1
        hidden_size = 4096

        source_code = """
import torch
import triton
import triton.language as tl
import cupy as cp
import math

# ============ Triton: RMSNorm ============
@triton.jit
def rms_norm_kernel(
    input_ptr, output_ptr, weight_ptr,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_idx = tl.program_id(0)

    # Compute mean of squares
    row_start = row_idx * n_cols
    mean_square = 0.0

    # Sum squares
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        mean_square += tl.sum(vals * vals)

    mean_square = mean_square / n_cols
    rms = tl.sqrt(mean_square + eps)

    # Normalize
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        normalized = (vals / rms) * weight
        tl.store(output_ptr + row_start + cols, normalized, mask=mask)

# ============ CUDA: SiLU Activation ============
silu_kernel = cp.RawKernel(r'''
extern "C" __global__
void silu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}
''', 'silu_kernel')

def run(hidden_states):
    '''
    Llama3-8B Decoder Layer

    Input: (batch_size, seq_len, hidden_size)
    Output: (batch_size, seq_len, hidden_size)

    Architecture:
    1. RMSNorm (Triton)
    2. Self-Attention with GQA (PyTorch)
    3. Residual
    4. RMSNorm (Triton)
    5. SwiGLU MLP with SiLU (CUDA)
    6. Residual
    '''
    batch_size, seq_len, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Llama3-8B hyperparameters
    num_heads = 32
    num_kv_heads = 8  # Grouped Query Attention
    head_dim = 128
    intermediate_size = 11008
    eps = 1e-6

    # ============ 1. Pre-Attention RMSNorm (Triton) ============
    residual = hidden_states

    # Flatten for RMSNorm: (batch * seq_len, hidden_size)
    hidden_flat = hidden_states.view(-1, hidden_size)
    n_rows = hidden_flat.shape[0]

    # RMSNorm weight (gamma)
    rms_weight_1 = torch.ones(hidden_size, device=device, dtype=dtype)
    normalized_1 = torch.empty_like(hidden_flat)

    grid = (n_rows,)
    rms_norm_kernel[grid](
        hidden_flat, normalized_1, rms_weight_1,
        hidden_size, eps,
        BLOCK_SIZE=256
    )

    hidden_states = normalized_1.view(batch_size, seq_len, hidden_size)

    # ============ 2. Self-Attention with GQA ============
    # QKV Projections
    W_q = torch.randn(hidden_size, num_heads * head_dim, device=device, dtype=dtype) * 0.02
    W_k = torch.randn(hidden_size, num_kv_heads * head_dim, device=device, dtype=dtype) * 0.02
    W_v = torch.randn(hidden_size, num_kv_heads * head_dim, device=device, dtype=dtype) * 0.02

    q = hidden_states @ W_q
    k = hidden_states @ W_k
    v = hidden_states @ W_v

    # Reshape: (batch, seq_len, num_heads, head_dim)
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

    # Expand KV for Grouped Query Attention
    kv_repeat = num_heads // num_kv_heads
    k = k.repeat_interleave(kv_repeat, dim=2)
    v = v.repeat_interleave(kv_repeat, dim=2)

    # Transpose for attention: (batch, num_heads, seq_len, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # Reshape: (batch, seq_len, hidden_size)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, hidden_size)

    # Output projection
    W_o = torch.randn(num_heads * head_dim, hidden_size, device=device, dtype=dtype) * 0.02
    attn_output = attn_output @ W_o

    # ============ 3. Residual Connection ============
    hidden_states = residual + attn_output

    # ============ 4. Pre-MLP RMSNorm (Triton) ============
    residual = hidden_states

    hidden_flat = hidden_states.view(-1, hidden_size)
    rms_weight_2 = torch.ones(hidden_size, device=device, dtype=dtype)
    normalized_2 = torch.empty_like(hidden_flat)

    rms_norm_kernel[grid](
        hidden_flat, normalized_2, rms_weight_2,
        hidden_size, eps,
        BLOCK_SIZE=256
    )

    hidden_states = normalized_2.view(batch_size, seq_len, hidden_size)

    # ============ 5. SwiGLU MLP ============
    # Gate and Up projections
    W_gate = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype) * 0.02
    W_up = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype) * 0.02

    gate = hidden_states @ W_gate
    up = hidden_states @ W_up

    # Apply SiLU to gate (CUDA)
    gate_silu = torch.empty_like(gate)
    n_elements = gate.numel()
    threads_per_block = 256
    blocks = (n_elements + threads_per_block - 1) // threads_per_block

    silu_kernel(
        (blocks,), (threads_per_block,),
        (cp.asarray(gate), cp.asarray(gate_silu), n_elements)
    )

    # Element-wise multiply (SwiGLU)
    mlp_output = gate_silu * up

    # Down projection
    W_down = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
    mlp_output = mlp_output @ W_down

    # ============ 6. Residual Connection ============
    output = residual + mlp_output

    return output
"""

        request = {
            "kernel": {
                "source_code": source_code,
                "kernel_type": "multi_kernel",
                "metadata": {
                    "entry_point": "run",
                    "description": f"Llama3-8B decoder layer - {workload_name} workload"
                },
                "io": {
                    "args": [
                        {
                            "name": "hidden_states",
                            "type": "tensor",
                            "tensor_spec": {
                                "shape": [batch_size, seq_len, hidden_size],
                                "dtype": "float32",
                                "init": {"kind": "randn", "seed": 42}
                            },
                            "role": "input"
                        }
                    ]
                }
            },
            "num_trials": 10,
            "timeout": 300  # 5 minutes - needed for NCU profiling with complex kernels
        }

        response = requests.post(f"{server_url}/evaluate", json=request)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        kernel_result = result.get("kernel_exec_result", {})
        metadata = kernel_result.get("metadata", {})

        # Verify compilation and execution
        assert kernel_result.get("compiled") == True, f"Kernel compilation failed: {kernel_result.get('compilation_error')}"

        # Verify runtime stats are present
        runtime_stats = kernel_result.get("runtime_stats", {})
        assert "mean" in runtime_stats, "Missing runtime statistics"
        assert runtime_stats["mean"] > 0, "Invalid runtime mean"

        # Verify device_metrics
        assert "device_metrics" in metadata
        device_metrics = metadata["device_metrics"]
        print(f"Device Metrics: \n{device_metrics}\n")

        # Print performance metrics for visibility
        print(f"\n{workload_name.upper()} workload (seq_len={seq_len}):")
        print(f"  Mean: {runtime_stats.get('mean', 0)*1000:.3f} ms")
        print(f"  Std:  {runtime_stats.get('std', 0)*1000:.3f} ms")
        print(f"  Min:  {runtime_stats.get('min', 0)*1000:.3f} ms")
        print(f"  Max:  {runtime_stats.get('max', 0)*1000:.3f} ms")
