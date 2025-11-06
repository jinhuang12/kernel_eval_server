#!/usr/bin/env python3
"""
Test script for simplified Triton compilation backend
"""

import torch
import logging
from compilation.triton.triton_backend import TritonCompilationBackend
from shared.models import (
    KernelCode, KernelType, IOContract, ArgSpec, TensorInit, TensorSpec, 
    LaunchConfig, LaunchDim
)
from io_contract import IOContractManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_vector_add():
    """Test a simple vector addition kernel"""
    
    # Triton kernel source code
    triton_source = """
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr, 
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

    # Define IOContract
    n_elements = 1024
    block_size = 128
    
    io_contract = IOContract(
        args=[
            # Input tensors
            ArgSpec(
                name="x_ptr",
                type="tensor",
                tensor_spec=TensorSpec(
                    shape=[n_elements],
                    dtype="float32"
                ),
                role="input"
            ),
            ArgSpec(
                name="y_ptr", 
                type="tensor",
                tensor_spec=TensorSpec(
                    shape=[n_elements],
                    dtype="float32"
                ),
                role="input"
            ),
            # Output tensor
            ArgSpec(
                name="output_ptr",
                type="tensor",
                tensor_spec=TensorSpec(
                    shape=[n_elements],
                    dtype="float32"
                ),
                role="output"
            ),
            # Scalar arguments
            ArgSpec(
                name="n_elements",
                type="int",
                value=n_elements
            ),
            # Constexpr parameter
            ArgSpec(
                name="BLOCK_SIZE",
                type="int",
                value=block_size,
                is_meta=True
            )
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=n_elements // block_size, y=1, z=1),
            num_warps=4
        )
    )
    
    # Create kernel code
    kernel_code = KernelCode(
        kernel_type=KernelType.TRITON,
        source_code=triton_source,
        io=io_contract
    )
    
    # Compile kernel
    backend = TritonCompilationBackend()
    executable = backend.compile(kernel_code, gpu_id=0)
    
    logger.info(f"Compiled kernel: {executable.kernel_name}")
    logger.info(f"Default inputs generated: {len(executable._default_inputs)}")
    
    # Test execution with generated inputs
    x = torch.randn(n_elements, device='cuda:0')
    y = torch.randn(n_elements, device='cuda:0')
    out = torch.empty(n_elements, device='cuda:0')
    
    # Execute kernel
    output = executable._execute_impl(x, y, out, n_elements, block_size)
    
    # Verify result
    expected = x + y
    if torch.allclose(output, expected, rtol=1e-5):
        logger.info("✓ Vector addition test passed!")
    else:
        logger.error("✗ Vector addition test failed!")
        logger.error(f"Max error: {(output - expected).abs().max().item()}")
    
    return executable


def test_matrix_multiply():
    """Test a matrix multiply kernel with more complex parameters"""
    
    triton_source = """
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, tl.cdiv(M - first_pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M))
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k)
        b_mask = (offs_k[:, None] < K - k) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float32)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)
"""
    
    # Matrix dimensions
    M, N, K = 512, 512, 512
    io_manager = IOContractManager()

    io_contract = IOContract(
        args=[
            ArgSpec(name="a_ptr", type="tensor", 
                   tensor_spec=TensorSpec(shape=[M, K], dtype="float32", init=TensorInit("full", fill_value=3.0)), role="input"),
            ArgSpec(name="b_ptr", type="tensor",
                   tensor_spec=TensorSpec(shape=[K, N], dtype="float32", init=TensorInit("full", fill_value=7.0)), role="input"),
            ArgSpec(name="c_ptr", type="tensor",
                   tensor_spec=TensorSpec(shape=[M, N], dtype="float32"), role="output"),
            ArgSpec(name="M", type="int", value=M),
            ArgSpec(name="N", type="int", value=N),
            ArgSpec(name="K", type="int", value=K),
            ArgSpec(name="stride_am", type="int", value=K),
            ArgSpec(name="stride_ak", type="int", value=1),
            ArgSpec(name="stride_bk", type="int", value=N),
            ArgSpec(name="stride_bn", type="int", value=1),
            ArgSpec(name="stride_cm", type="int", value=N),
            ArgSpec(name="stride_cn", type="int", value=1),
            ArgSpec(name="BLOCK_SIZE_M", type="int", value=64, is_meta=True),
            ArgSpec(name="BLOCK_SIZE_N", type="int", value=64, is_meta=True),
            ArgSpec(name="BLOCK_SIZE_K", type="int", value=32, is_meta=True),
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=(M // 64) * (N // 64), y=1, z=1),
            num_warps=8,
            num_stages=3
        )
    )
    
    kernel_code = KernelCode(
        kernel_type=KernelType.TRITON,
        source_code=triton_source,
        io=io_contract
    )
    
    backend = TritonCompilationBackend()
    executable = backend.compile(kernel_code, gpu_id=0)
    
    logger.info(f"Compiled matmul kernel: {executable.kernel_name}")
    
    # Test with specific inputs
    inputs = io_manager.generate_inputs(io_contract, torch.device('cuda:0'))
    
    # Execute kernel with explicit inputs (overriding defaults)
    c = executable(*inputs)
    
    # Verify
    expected = torch.matmul(inputs[0], inputs[1])
    if torch.allclose(c, expected, rtol=1e-3, atol=1e-3):
        logger.info("✓ Matrix multiply test passed!")
    else:
        max_error = (c - expected).abs().max().item()
        logger.error(f"✗ Matrix multiply test failed! Max error: {max_error}")
    
    return executable


if __name__ == "__main__":
    logger.info("Testing simplified Triton compilation backend...")
    logger.info("=" * 60)
    
    try:
        logger.info("\n1. Testing vector addition kernel...")
        test_simple_vector_add()
        
        logger.info("\n2. Testing matrix multiply kernel...")
        test_matrix_multiply()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()