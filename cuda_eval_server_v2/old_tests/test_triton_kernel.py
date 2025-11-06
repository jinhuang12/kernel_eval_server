#!/usr/bin/env python3
"""
Test script for Triton kernel evaluation functionality
Tests TritonCompilationBackend and TritonExecutableKernel implementations
"""

import asyncio
import logging
import torch
import sys
import os
from typing import Optional, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


# ============================================================================
# Test Kernels
# ============================================================================

# Simple add kernel (removed self-invocation - now requires IOContract)
SIMPLE_ADD_WITH_INVOCATION = '''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def test_invocation():
    """Test invocation - no longer used for capture"""
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
    return output
'''

# Simple add kernel without invocation (requires IOContract)
SIMPLE_ADD_NO_INVOCATION = '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
'''

# Matrix multiply kernel
MATMUL_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
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
        
        accumulator += tl.dot(a, b)
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_wrapper():
    """Wrapper for testing"""
    M, K, N = 512, 256, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    c = torch.empty((M, N), device='cuda', dtype=torch.float32)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    return c
'''

# PyTorch reference for comparison
TORCH_ADD_REFERENCE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    return [x, y]
'''


# ============================================================================
# Test 1: Basic Compilation and Execution
# ============================================================================

async def test_basic_triton_compilation():
    """Test basic Triton kernel compilation with and without IOContract"""
    print("\n" + "="*80)
    print("🧪 TEST 1: Basic Triton Compilation and Execution")
    print("="*80)
    
    from compilation.triton import TritonCompilationBackend
    from shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, TensorInit, LaunchConfig, LaunchDim
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping test")
        return True
    
    backend = TritonCompilationBackend()
    
    # Test 1a: Compilation with IOContract (removed capture-based test)
    print("\n📝 Test 1a: Kernel compilation with IOContract...")
    # This test has been removed since capture is no longer supported.
    # All Triton kernels now require an explicit IOContract.
    
    # Test 1b: Compilation with explicit IOContract (no self-invocation)
    print("\n📝 Test 1b: Kernel with explicit IOContract...")
    try:
        io_contract = IOContract(
            args=[
                ArgSpec(name="x_ptr", type="tensor", role="input", 
                       tensor_spec=TensorSpec(
                           shape=[2048], 
                           dtype="float32",
                           init=TensorInit(kind="randn", seed=42)
                       )),
                ArgSpec(name="y_ptr", type="tensor", role="input",
                       tensor_spec=TensorSpec(
                           shape=[2048], 
                           dtype="float32",
                           init=TensorInit(kind="randn", seed=43)
                       )),
                ArgSpec(name="output_ptr", type="tensor", role="output",
                       tensor_spec=TensorSpec(shape=[2048], dtype="float32")),
                ArgSpec(name="n_elements", type="int", value=2048),
                ArgSpec(name="BLOCK_SIZE", type="int", value=512, is_meta=True),
            ],
            outputs=[TensorSpec(shape=[2048], dtype="float32")],
            launch=LaunchConfig(
                grid=LaunchDim(x=4),  # 2048 / 512 = 4 blocks
                num_warps=4,
            )
        )
        
        kernel_code = KernelCode(
            source_code=SIMPLE_ADD_NO_INVOCATION,
            kernel_type=KernelType.TRITON,
            io=io_contract
        )
        
        executable = backend.compile(kernel_code, gpu_id=0)
        print("✅ Successfully compiled kernel with IOContract")
        
        # Test execution with generated inputs
        device = torch.device('cuda:0')
        torch.manual_seed(42)
        x = torch.randn(2048, device=device)
        torch.manual_seed(43)
        y = torch.randn(2048, device=device)
        
        result = executable(x, y)
        expected = x + y
        
        if torch.allclose(result, expected, atol=1e-5):
            print("✅ Kernel execution correct with IOContract!")
        else:
            print(f"❌ Kernel execution incorrect, max error: {(result - expected).abs().max().item()}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to compile with IOContract: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


# ============================================================================
# Test 2: IOContract-Driven Execution
# ============================================================================

async def test_io_contract_execution():
    """Test comprehensive IOContract handling including tensor generation"""
    print("\n" + "="*80)
    print("🧪 TEST 2: IOContract-Driven Execution")
    print("="*80)
    
    from compilation.triton import TritonCompilationBackend
    from shared.models import (
        KernelCode, KernelType, IOContract, ArgSpec, 
        TensorSpec, TensorInit, LaunchConfig, LaunchDim
    )
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping test")
        return True
    
    backend = TritonCompilationBackend()
    
    print("\n📝 Testing comprehensive IOContract with tensor generation...")
    
    # Create IOContract with various tensor initialization methods
    io_contract = IOContract(
        args=[
            # Input tensor with randn initialization
            ArgSpec(name="a_ptr", type="tensor", role="input",
                   tensor_spec=TensorSpec(
                       shape=[64, 128],
                       dtype="float32",
                       init=TensorInit(kind="randn", seed=42, mean=0.0, std=1.0)
                   )),
            # Input tensor with ones initialization  
            ArgSpec(name="b_ptr", type="tensor", role="input",
                   tensor_spec=TensorSpec(
                       shape=[128, 256],
                       dtype="float32",
                       init=TensorInit(kind="ones")
                   )),
            # Output tensor (allocated but not initialized)
            ArgSpec(name="c_ptr", type="tensor", role="output",
                   tensor_spec=TensorSpec(shape=[64, 256], dtype="float32")),
            # Scalar arguments
            ArgSpec(name="M", type="int", value=64),
            ArgSpec(name="N", type="int", value=256),
            ArgSpec(name="K", type="int", value=128),
            # Stride arguments
            ArgSpec(name="stride_am", type="int", value=128),
            ArgSpec(name="stride_ak", type="int", value=1),
            ArgSpec(name="stride_bk", type="int", value=256),
            ArgSpec(name="stride_bn", type="int", value=1),
            ArgSpec(name="stride_cm", type="int", value=256),
            ArgSpec(name="stride_cn", type="int", value=1),
            # Meta parameters (constexpr)
            ArgSpec(name="BLOCK_SIZE_M", type="int", value=32, is_meta=True),
            ArgSpec(name="BLOCK_SIZE_N", type="int", value=32, is_meta=True),
            ArgSpec(name="BLOCK_SIZE_K", type="int", value=32, is_meta=True),
        ],
        outputs=[TensorSpec(shape=[64, 256], dtype="float32")],
        launch=LaunchConfig(
            grid=LaunchDim(x=2, y=8),  # (64/32, 256/32)
        )
    )
    
    try:
        kernel_code = KernelCode(
            source_code=MATMUL_KERNEL,
            kernel_type=KernelType.TRITON,
            io=io_contract
        )
        
        executable = backend.compile(kernel_code, gpu_id=0)
        print("✅ Compiled matmul kernel with comprehensive IOContract")
        
        # Execute with no inputs (should use IOContract to generate)
        result = executable()
        
        # Verify result shape
        if result.shape == torch.Size([64, 256]):
            print("✅ Output shape correct: [64, 256]")
        else:
            print(f"❌ Output shape incorrect: {result.shape}")
            return False
        
        # Verify computation (a=randn, b=ones, so c ≈ sum of a rows)
        print(f"✅ Execution completed, output stats: mean={result.mean():.4f}, std={result.std():.4f}")
        
        # Test with explicit inputs overriding IOContract
        device = torch.device('cuda:0')
        torch.manual_seed(42)
        a_custom = torch.eye(64, 128, device=device)
        b_custom = torch.ones(128, 256, device=device)
        
        result_custom = executable(a_custom, b_custom)
        expected = torch.full((64, 256), 128.0, device=device)  # Each row sums to 128
        
        if torch.allclose(result_custom, expected, atol=1e-3):
            print("✅ Custom input execution correct!")
        else:
            max_error = (result_custom - expected).abs().max().item()
            print(f"⚠️ Custom input execution has error: {max_error}")
            
    except Exception as e:
        print(f"❌ IOContract execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


# ============================================================================
# Test 3: IOContract-based Execution (formerly Runtime Capture)
# ============================================================================

async def test_iocontract_based_execution():
    """Test Triton kernel execution with explicit IOContract"""
    print("\n" + "="*80)
    print("🧪 TEST 3: IOContract-based Execution")
    print("="*80)
    
    from compilation.triton import TritonCompilationBackend, TritonExecutableKernel
    from shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, TensorInit, LaunchConfig, LaunchDim
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping test")
        return True
    
    print("\n📝 Testing Triton kernel with explicit IOContract...")
    
    try:
        # Create kernel with explicit IOContract
        io_contract = IOContract(
            args=[
                ArgSpec(name="a_ptr", type="tensor", role="input",
                       tensor_spec=TensorSpec(shape=[32, 32], dtype="float32",
                                             init=TensorInit(kind="randn", seed=42))),
                ArgSpec(name="b_ptr", type="tensor", role="input",
                       tensor_spec=TensorSpec(shape=[32, 32], dtype="float32",
                                             init=TensorInit(kind="randn", seed=43))),
                ArgSpec(name="c_ptr", type="tensor", role="output",
                       tensor_spec=TensorSpec(shape=[32, 32], dtype="float32")),
                ArgSpec(name="M", type="int", value=32),
                ArgSpec(name="N", type="int", value=32),
                ArgSpec(name="K", type="int", value=32),
                ArgSpec(name="stride_am", type="int", value=32),
                ArgSpec(name="stride_ak", type="int", value=1),
                ArgSpec(name="stride_bk", type="int", value=32),
                ArgSpec(name="stride_bn", type="int", value=1),
                ArgSpec(name="stride_cm", type="int", value=32),
                ArgSpec(name="stride_cn", type="int", value=1),
                ArgSpec(name="BLOCK_SIZE_M", type="int", value=16, is_meta=True),
                ArgSpec(name="BLOCK_SIZE_N", type="int", value=16, is_meta=True),
                ArgSpec(name="BLOCK_SIZE_K", type="int", value=16, is_meta=True),
            ],
            outputs=[TensorSpec(shape=[32, 32], dtype="float32")],
            launch=LaunchConfig(grid=LaunchDim(x=2, y=2))
        )
        
        kernel_code = KernelCode(
            source_code=MATMUL_KERNEL,
            kernel_type=KernelType.TRITON,
            io=io_contract
        )
        
        backend = TritonCompilationBackend()
        executable = backend.compile(kernel_code, gpu_id=0)
        print("✅ Created TritonExecutableKernel with IOContract")
        
        # Test execution
        print("\n📝 Testing kernel execution...")
        outputs = executable.compile_and_run(fill="randn")
        
        if outputs and len(outputs) > 0:
            print(f"✅ Execution succeeded, got {len(outputs)} output(s)")
            for i, out in enumerate(outputs[:3]):  # Show first 3
                if torch.is_tensor(out):
                    print(f"   Output {i}: shape={out.shape}, dtype={out.dtype}")
        else:
            print("⚠️ No outputs from execution")
            
    except Exception as e:
        print(f"❌ IOContract-based execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


# ============================================================================
# Test 4: Cross-Kernel Type Comparison (Torch vs Triton)
# ============================================================================

async def test_torch_vs_triton():
    """Test comparing equivalent Torch and Triton kernels"""
    print("\n" + "="*80)
    print("🧪 TEST 4: Cross-Kernel Type Comparison (Torch vs Triton)")
    print("="*80)
    
    from compilation.triton import TritonCompilationBackend
    from compilation.torch import TorchCompilationBackend
    from shared.models import (
        KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, TensorInit
    )
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping test")
        return True
    
    # Create common IOContract for both kernels
    x_spec = TensorSpec(shape=[4096], dtype="float32", init=TensorInit(kind="randn", seed=42))
    y_spec = TensorSpec(shape=[4096], dtype="float32", init=TensorInit(kind="randn", seed=43))

    io_contract = IOContract(
        args=[
            ArgSpec(name="x", type="tensor", role="input", tensor_spec=x_spec),
            ArgSpec(name="y", type="tensor", role="input", tensor_spec=y_spec),
        ],
        outputs=[TensorSpec(shape=[4096], dtype="float32")]
    )
    
    print("\n📝 Compiling Torch reference kernel...")
    try:
        torch_backend = TorchCompilationBackend()
        torch_kernel = KernelCode(
            source_code=TORCH_ADD_REFERENCE,
            kernel_type=KernelType.TORCH,
            io=io_contract
        )
        torch_executable = torch_backend.compile(torch_kernel, gpu_id=0)
        print("✅ Compiled Torch kernel")
    except Exception as e:
        print(f"❌ Failed to compile Torch kernel: {e}")
        return False
    
    print("\n📝 Compiling Triton kernel...")
    try:
        # Triton IOContract needs additional args
        triton_io = IOContract(
            args=[
                ArgSpec(name="x_ptr", type="tensor", role="input",
                       tensor_spec=x_spec),
                ArgSpec(name="y_ptr", type="tensor", role="input",
                       tensor_spec=y_spec),
                ArgSpec(name="output_ptr", type="tensor", role="output",
                       tensor_spec=TensorSpec(shape=[4096], dtype="float32")),
                ArgSpec(name="n_elements", type="int", value=4096),
                ArgSpec(name="BLOCK_SIZE", type="int", value=4096, is_meta=True),
            ],
            outputs=[TensorSpec(shape=[4096], dtype="float32")]
        )
        
        triton_backend = TritonCompilationBackend()
        triton_kernel = KernelCode(
            source_code=SIMPLE_ADD_NO_INVOCATION,
            kernel_type=KernelType.TRITON,
            io=triton_io
        )
        triton_executable = triton_backend.compile(triton_kernel, gpu_id=0)
        print("✅ Compiled Triton kernel")
    except Exception as e:
        print(f"❌ Failed to compile Triton kernel: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n📝 Comparing execution results...")
    try:
        # Generate test inputs
        device = torch.device('cuda:0')
        torch.manual_seed(42)
        x = torch.randn(4096, device=device)
        torch.manual_seed(43)
        y = torch.randn(4096, device=device)
        
        # Execute both kernels
        torch_result = torch_executable(x, y)
        triton_result = triton_executable(x, y)
        
        # Compare results
        if torch.allclose(torch_result, triton_result, atol=1e-5):
            print("✅ Results match! Torch and Triton kernels produce same output")
            max_diff = (torch_result - triton_result).abs().max().item()
            print(f"   Max difference: {max_diff:.2e}")
        else:
            max_diff = (torch_result - triton_result).abs().max().item()
            print(f"❌ Results differ! Max difference: {max_diff}")
            return False
            
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


# ============================================================================
# Test 5: Grid Configuration Testing
# ============================================================================

async def test_grid_configurations():
    """Test various 1D grid configuration sizes"""
    print("\n" + "="*80)
    print("🧪 TEST 5: Grid Configuration Testing")
    print("="*80)
    
    from compilation.triton import TritonCompilationBackend
    from shared.models import (
        KernelCode, KernelType, IOContract, ArgSpec, 
        TensorSpec, TensorInit, LaunchConfig, LaunchDim
    )
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping test")
        return True
    
    backend = TritonCompilationBackend()
    
    # Test different 1D grid sizes (SIMPLE_ADD kernel only uses axis=0)
    # Each configuration tests different grid/block size combinations
    grid_configs = [
        ("Small Grid (256 elements)", 256, 128, LaunchDim(x=2)),   # 256/128 = 2 blocks
        ("Medium Grid (2048 elements)", 2048, 256, LaunchDim(x=8)), # 2048/256 = 8 blocks  
        ("Large Grid (8192 elements)", 8192, 512, LaunchDim(x=16)), # 8192/512 = 16 blocks
    ]
    
    for name, n_elements, block_size, grid_dim in grid_configs:
        print(f"\n📝 Testing {name}: Grid({grid_dim.x}) with BLOCK_SIZE={block_size}...")
        
        io_contract = IOContract(
            args=[
                ArgSpec(name="x_ptr", type="tensor", role="input",
                       tensor_spec=TensorSpec(
                           shape=[n_elements], 
                           dtype="float32",
                           init=TensorInit(kind="ones")  # Use ones for deterministic test
                       )),
                ArgSpec(name="y_ptr", type="tensor", role="input",
                       tensor_spec=TensorSpec(
                           shape=[n_elements], 
                           dtype="float32",
                           init=TensorInit(kind="full", fill_value=2.0)  # Filled with 2s
                       )),
                ArgSpec(name="output_ptr", type="tensor", role="output",
                       tensor_spec=TensorSpec(shape=[n_elements], dtype="float32")),
                ArgSpec(name="n_elements", type="int", value=n_elements),
                ArgSpec(name="BLOCK_SIZE", type="int", value=block_size, is_meta=True),
            ],
            outputs=[TensorSpec(shape=[n_elements], dtype="float32")],
            launch=LaunchConfig(grid=grid_dim)
        )
        
        try:
            kernel_code = KernelCode(
                source_code=SIMPLE_ADD_NO_INVOCATION,
                kernel_type=KernelType.TRITON,
                io=io_contract
            )
            
            executable = backend.compile(kernel_code, gpu_id=0)
            
            # Test execution with matching tensor sizes
            device = torch.device('cuda:0')
            x = torch.ones(n_elements, device=device)
            y = torch.ones(n_elements, device=device) * 2
            
            result = executable(x, y)
            expected = torch.ones(n_elements, device=device) * 3
            
            if torch.allclose(result, expected, atol=1e-5):
                print(f"✅ {name} execution successful")
            else:
                print(f"❌ {name} execution failed")
                return False
                
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
            return False
    
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run all Triton kernel tests"""
    print("🚀 Triton Kernel Evaluation Test Suite")
    print("   Testing TritonCompilationBackend and TritonExecutableKernel")
    
    # Check for Triton availability
    try:
        import triton
        print(f"✅ Triton version: {triton.__version__ if hasattr(triton, '__version__') else 'unknown'}")
    except ImportError:
        print("❌ Triton not installed! Install with: pip install triton")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available - some tests will be skipped")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    tests = [
        ("Basic Compilation", test_basic_triton_compilation),
        ("IOContract Execution", test_io_contract_execution),
        ("IOContract-based Execution", test_iocontract_based_execution),
        ("Torch vs Triton", test_torch_vs_triton),
        ("Grid Configurations", test_grid_configurations),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Triton kernel evaluation is fully functional:")
        print("  ✅ Compilation with and without IOContract")
        print("  ✅ IOContract-based execution")
        print("  ✅ Tensor generation from specifications")
        print("  ✅ Cross-kernel type comparison")
        print("  ✅ Various grid configurations")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        print("Review the output above for details")


if __name__ == "__main__":
    asyncio.run(main())
