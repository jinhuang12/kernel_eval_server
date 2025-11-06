"""
Test script for CUDA backend with multiple kernels and compiler options
"""

import json
import logging
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compilation.cuda.cuda_backend import CudaCompilationBackend
from shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, LaunchConfig, LaunchDim, TensorInit
from shared.kernel_metadata import CudaKernelMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_multi_kernel_source():
    """Create CUDA source with multiple kernels"""
    
    cuda_source = """
// Vector operations kernels
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void vector_mul(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

__global__ void vector_scale(float* input, float* output, float scale, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid] * scale;
    }
}

// Reduction kernel
__global__ void vector_sum(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Simple reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
"""
    return cuda_source


def test_single_kernel_auto_entrypoint():
    """Test single kernel with automatic entrypoint detection"""
    print("\n" + "="*60)
    print("Test: Single Kernel with Auto Entrypoint")
    print("="*60)
    
    # Single kernel source
    cuda_source = """
__global__ void simple_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] * 2.0f;
    }
}
"""
    
    # No need to specify kernel_name for single kernel
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        io=IOContract(
            args=[
                ArgSpec(name="data", type="tensor", role="inout",
                       tensor_spec=TensorSpec(shape=[1024], dtype="float32",
                                             init=TensorInit(kind="ones"))),
                ArgSpec(name="n", type="int", value=1024)
            ],
            launch=LaunchConfig(
                grid=LaunchDim(x=4, y=1, z=1),
                block=LaunchDim(x=256, y=1, z=1)
            )
        )
    )
    
    backend = CudaCompilationBackend()
    
    try:
        executable = backend.compile(kernel_code, gpu_id=0)
        print(f"✅ Compilation successful")
        print(f"   Entrypoint: {executable.entrypoint_name}")
        
        # Execute
        result = executable()
        if torch.is_tensor(result):
            print(f"✅ Execution successful")
            print(f"   Result: first 5 values = {result[:5].tolist()}")
    except Exception as e:
        print(f"❌ Failed: {e}")


def test_multiple_kernels():
    """Test multiple kernels with explicit entrypoint"""
    print("\n" + "="*60)
    print("Test: Multiple Kernels with Entrypoint Selection")
    print("="*60)
    
    cuda_source = create_multi_kernel_source()
    n = 1024
    
    # Create metadata with explicit entrypoint
    metadata = CudaKernelMetadata(
        kernel_name="vector_add",  # Specify entrypoint
        compiler_options=["--std=c++14"]    # Valid NVRTC option
    )
    
    # IOContract for vector_add
    io_contract = IOContract(
        args=[
            ArgSpec(name="a", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32",
                                         init=TensorInit(kind="ones"))),
            ArgSpec(name="b", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32",
                                         init=TensorInit(kind="ones"))),
            ArgSpec(name="c", type="tensor", role="output",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32")),
            ArgSpec(name="n", type="int", value=n)
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=4, y=1, z=1),
            block=LaunchDim(x=256, y=1, z=1)
        )
    )
    
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        metadata=metadata,
        io=io_contract
    )
    
    backend = CudaCompilationBackend()
    
    try:
        executable = backend.compile(kernel_code, gpu_id=0)
        print(f"✅ Compilation successful")
        print(f"   Default entrypoint: {executable.entrypoint_name}")
        
        # Execute default kernel (vector_add)
        result = executable()
        if torch.is_tensor(result):
            print(f"✅ vector_add execution successful")
            print(f"   Result: first 5 values = {result[:5].tolist()}")
    except Exception as e:
        print(f"❌ Failed: {e}")


def test_kernel_switching():
    """Test switching between kernels at runtime"""
    print("\n" + "="*60)
    print("Test: Runtime Kernel Switching")
    print("="*60)
    
    cuda_source = create_multi_kernel_source()
    n = 1024
    
    # Create with vector_mul as entrypoint
    metadata = CudaKernelMetadata(kernel_name="vector_mul")
    
    # IOContract compatible with both vector_add and vector_mul
    io_contract = IOContract(
        args=[
            ArgSpec(name="a", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32",
                                         init=TensorInit(kind="full", fill_value=2.0))),
            ArgSpec(name="b", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32",
                                         init=TensorInit(kind="full", fill_value=3.0))),
            ArgSpec(name="c", type="tensor", role="output",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32")),
            ArgSpec(name="n", type="int", value=n)
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=4, y=1, z=1),
            block=LaunchDim(x=256, y=1, z=1)
        )
    )
    
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        metadata=metadata,
        io=io_contract
    )
    
    backend = CudaCompilationBackend()
    
    try:
        executable = backend.compile(kernel_code, gpu_id=0)
        print(f"✅ Compilation successful")
        print(f"   Initial entrypoint: {executable.entrypoint_name}")
        
        # Execute vector_mul (2 * 3 = 6)
        result = executable()
        if torch.is_tensor(result):
            print(f"✅ vector_mul execution: {result[0].item():.1f} (expected 6.0)")
        
        # Switch to vector_add
        executable.set_entrypoint("vector_add")
        print(f"   Switched to: {executable.entrypoint_name}")
        
        # Execute vector_add (2 + 3 = 5)
        result = executable()
        if torch.is_tensor(result):
            print(f"✅ vector_add execution: {result[0].item():.1f} (expected 5.0)")
        
        # Use execute_kernel for temporary execution
        result = executable.execute_kernel("vector_mul")
        if torch.is_tensor(result):
            print(f"✅ Temporary vector_mul: {result[0].item():.1f} (expected 6.0)")
        
        # Verify we're back to vector_add
        print(f"   Current entrypoint: {executable.entrypoint_name}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()


def test_compiler_options():
    """Test compiler options and backend selection"""
    print("\n" + "="*60)
    print("Test: Compiler Options")
    print("="*60)
    
    cuda_source = """
__global__ void optimized_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Should be optimized with -use_fast_math
        data[tid] = sqrtf(data[tid]) * 2.0f;
    }
}
"""
    
    # Test with valid NVRTC options
    metadata = CudaKernelMetadata(
        kernel_name="optimized_kernel",
        compiler_options=["--use_fast_math", "--std=c++14"],
        backend="nvrtc"  # Explicitly use NVRTC backend
    )
    
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        metadata=metadata,
        io=IOContract(
            args=[
                ArgSpec(name="data", type="tensor", role="inout",
                       tensor_spec=TensorSpec(shape=[1024], dtype="float32",
                                             init=TensorInit(kind="full", fill_value=4.0))),
                ArgSpec(name="n", type="int", value=1024)
            ],
            launch=LaunchConfig(
                grid=LaunchDim(x=4, y=1, z=1),
                block=LaunchDim(x=256, y=1, z=1)
            )
        )
    )
    
    backend = CudaCompilationBackend()
    
    try:
        executable = backend.compile(kernel_code, gpu_id=0)
        print(f"✅ Compilation with options successful")
        print(f"   Options: {metadata.compiler_options}")
        print(f"   Backend: {metadata.backend}")
        
        result = executable()
        if torch.is_tensor(result):
            print(f"✅ Execution successful")
            print(f"   Result: sqrt(4) * 2 = {result[0].item():.1f}")
    except Exception as e:
        print(f"❌ Failed: {e}")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("CUDA Multi-Kernel Test Suite")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping tests")
        return
    
    tests = [
        test_single_kernel_auto_entrypoint,
        test_multiple_kernels,
        test_kernel_switching,
        test_compiler_options
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)


if __name__ == "__main__":
    main()