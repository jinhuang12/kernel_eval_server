"""
Test script for CUDA backend with example kernels
"""

import json
import logging
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from KernelBench.scripts.cuda_eval_server_v2.compilation.cuda.cuda_backend import CudaCompilationBackend
from KernelBench.scripts.cuda_eval_server_v2.shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, LaunchConfig, LaunchDim, TensorInit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_vector_add_kernel():
    """Create a simple vector addition CUDA kernel with IOContract"""
    
    # CUDA kernel source
    cuda_source = """
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(const float* diag, const float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int row = idx / M;
        int col = idx % M;
        C[idx] = diag[row] * B[row * M + col];
    }
}
"""
    
    # Create IOContract
    n = 1024
    io_contract = IOContract.from_dict({
    "args": [
        {
            "name": "A",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [4096, 4096],
                "dtype": "float32",
                "init": {"kind": "full", "fill_value": 2.0}
            }
        },
        {
            "name": "B", 
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [4096, 4096],
                "dtype": "float32", 
                "init": {"kind": "full", "fill_value": 3.0}
            }
        },
        {
            "name": "C",
            "type": "tensor",
            "role": "output",
            "tensor_spec": {
                "shape": [4096, 4096],
                "dtype": "float32"
            }
        },
        {"name": "N", "type": "int", "value": 16777216, "role": "input"},
        {"name": "M", "type": "int", "value": 4096, "role": "input"},
    ],
    "launch": {
        "grid": {"x": 65536, "y": 1, "z": 1},
        "block": {"x": 256, "y": 1, "z": 1}
    }
})
    
    # Create KernelCode
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        io=io_contract,
        #metadata={"kernel_name": "batched_matmul_kernel"}
    )
    
    return kernel_code


def create_matrix_multiply_kernel():
    """Create a simple matrix multiplication CUDA kernel"""
    
    cuda_source = """
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""
    
    # Matrix dimensions
    M, N, K = 64, 64, 64
    
    io_contract = IOContract(
        args=[
            ArgSpec(name="A", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[M, K], dtype="float32",
                                         init=TensorInit(kind="randn", seed=42))),
            ArgSpec(name="B", type="tensor", role="input", 
                   tensor_spec=TensorSpec(shape=[K, N], dtype="float32",
                                         init=TensorInit(kind="randn", seed=43))),
            ArgSpec(name="C", type="tensor", role="output",
                   tensor_spec=TensorSpec(shape=[M, N], dtype="float32")),
            ArgSpec(name="M", type="int", value=M),
            ArgSpec(name="N", type="int", value=N),
            ArgSpec(name="K", type="int", value=K)
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=4, y=4, z=1),
            block=LaunchDim(x=16, y=16, z=1)
        )
    )
    
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        io=io_contract,
        metadata={"kernel_name": "matrix_multiply"}
    )
    
    return kernel_code


def create_elementwise_kernel():
    """Create an elementwise operation kernel with multiple operations"""
    
    cuda_source = """
__global__ void elementwise_ops(float* input, float* output, float alpha, float beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // Apply operations: output = alpha * sin(val) + beta * cos(val)
        output[idx] = alpha * sinf(val) + beta * cosf(val);
    }
}
"""
    
    n = 4096
    io_contract = IOContract(
        args=[
            ArgSpec(name="input", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32",
                                         init=TensorInit(kind="uniform", low=0.0, high=6.28))),
            ArgSpec(name="output", type="tensor", role="output",
                   tensor_spec=TensorSpec(shape=[n], dtype="float32")),
            ArgSpec(name="alpha", type="float", value=0.5),
            ArgSpec(name="beta", type="float", value=0.5),
            ArgSpec(name="n", type="int", value=n)
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=16, y=1, z=1),
            block=LaunchDim(x=256, y=1, z=1)
        )
    )
    
    kernel_code = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        io=io_contract,
        metadata={"kernel_name": "elementwise_ops"}
    )
    
    return kernel_code


def test_compilation(kernel_code, kernel_name):
    """Test compilation of a CUDA kernel"""
    print(f"\n{'='*60}")
    print(f"Testing {kernel_name}")
    print('='*60)
    
    # Create backend
    backend = CudaCompilationBackend()
    
    try:
        # Compile kernel
        print(f"Compiling {kernel_name}...")
        executable_kernel = backend.compile(kernel_code, gpu_id=0)
        print(f"✅ Compilation successful!")
        
        # Check if it's executable
        if hasattr(executable_kernel, '_execute_impl'):
            print(f"✅ Kernel is executable")
        
        # Check kernel name
        if hasattr(executable_kernel, 'kernel_name'):
            print(f"✅ Kernel name: {executable_kernel.kernel_name}")
        
        # Check IOContract was passed
        if executable_kernel.io_contract:
            print(f"✅ IOContract preserved")
            print(f"   - {len(executable_kernel.io_contract.args)} arguments")
            print(f"   - Launch config: grid={executable_kernel.io_contract.launch.grid.x if executable_kernel.io_contract.launch else 'default'}")
        
        return executable_kernel
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_execution(executable_kernel, kernel_name):
    """Test execution of a compiled CUDA kernel"""
    print(f"\n{'='*60}")
    print(f"Testing execution of {kernel_name}")
    print('='*60)
    
    if not executable_kernel:
        print("❌ No executable kernel to test")
        return
    
    try:
        # Test with auto-generated inputs from IOContract
        print("Executing with IOContract-generated inputs...")
        result = executable_kernel()
        
        if result is not None:
            if torch.is_tensor(result):
                print(f"✅ Execution successful! Output shape: {result.shape}, dtype: {result.dtype}")
                print(f"   Output stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
            elif isinstance(result, tuple):
                print(f"✅ Execution successful! {len(result)} outputs")
                for i, out in enumerate(result):
                    if torch.is_tensor(out):
                        print(f"   Output {i}: shape={out.shape}, dtype={out.dtype}")
        else:
            print("✅ Execution successful! (no explicit outputs)")
        
        # Test with custom inputs if vector_add
        if kernel_name == "vector_add":
            print("\nTesting with custom inputs...")
            a = torch.ones([4096, 4096], device='cuda:0') * 2
            b = torch.ones([4096, 4096], device='cuda:0') * 3
            c = torch.zeros([4096, 4096], device='cuda:0') 
            
            result = executable_kernel(a, b, c, 16777216, 4096)
            
            # Check result
            expected = torch.bmm(a, b)
            if torch.allclose(c, expected, rtol=1e-5):
                print(f"✅ Correctness test passed! Result matches expected")
            else:
                print(f"❌ Correctness test failed!")
                print(f"   Expected: {expected[:5]}")
                print(f"   Got: {c[:5]}")
    
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("CUDA Backend Test Suite")
    print("="*60)
    
    # Test kernels
    test_cases = [
        ("vector_add", create_vector_add_kernel()),
        ("matrix_multiply", create_matrix_multiply_kernel()),
        ("elementwise_ops", create_elementwise_kernel())
    ]
    
    compiled_kernels = []
    
    # Test compilation
    for name, kernel_code in test_cases:
        executable = test_compilation(kernel_code, name)
        if executable:
            compiled_kernels.append((name, executable))
    
    # Test execution (only if CUDA is available)
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("EXECUTION TESTS")
        print("="*60)
        
        for name, executable in compiled_kernels:
            test_execution(executable, name)
    else:
        print("\n⚠️ CUDA not available on this machine, skipping execution tests")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Compiled: {len(compiled_kernels)}/{len(test_cases)} kernels")
    
    if compiled_kernels:
        print("\nSuccessfully compiled:")
        for name, _ in compiled_kernels:
            print(f"  - {name}")


if __name__ == "__main__":
    main()