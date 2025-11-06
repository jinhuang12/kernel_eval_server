"""
Simple PyTorch kernel evaluation example
"""

from kernel_eval_client import (
    KernelEvalClient,
    KernelCode,
    KernelType,
    IOContractBuilder,
    create_randn_spec
)


def main():
    # Create client
    client = KernelEvalClient("http://localhost:8000")
    
    # Define reference PyTorch kernel
    ref_kernel = KernelCode(
        source_code="""
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)
""",
        kernel_type=KernelType.TORCH
    )
    
    # Define optimized kernel (PyTorch with CUDA)
    custom_kernel = KernelCode(
        source_code="""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        cuda_source = '''
        __global__ void relu_kernel(float* x, int n) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < n) {
                x[tid] = fmaxf(0.0f, x[tid]);
            }
        }
        '''
        # Load inline CUDA kernel
        self.cuda_module = load_inline(
            name='relu_cuda',
            cpp_sources=[''],
            cuda_sources=[cuda_source],
            functions=['relu_kernel']
        )
    
    def forward(self, x):
        n = x.numel()
        grid = (n + 255) // 256
        self.cuda_module.relu_kernel(
            grid=(grid,), block=(256,),
            args=[x.data_ptr(), n]
        )
        return x
""",
        kernel_type=KernelType.TORCH_CUDA
    )
    
    # Optional: Specify input/output contract
    # If not specified, server will auto-generate inputs
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("x", create_randn_spec([1024, 1024], seed=42))
        .build()
    )
    
    # Add IOContract to kernels
    ref_kernel.io = io_contract
    custom_kernel.io = io_contract
    
    # Compare kernels
    print("Comparing PyTorch ReLU implementations...")
    result = client.compare(ref_kernel, custom_kernel, num_trials=100)
    
    # Check results
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        
        # Check compilation
        if exec_result["compiled"]:
            print("✓ Custom kernel compiled successfully")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
            return
        
        # Check correctness
        if exec_result["correctness"]:
            print("✓ Custom kernel passed validation")
        else:
            print(f"✗ Validation failed: {exec_result.get('validation_error')}")
            return
        
        # Show performance
        ref_runtime = result["ref_runtime"]["mean"]
        custom_runtime = exec_result["runtime"]
        speedup = ref_runtime / custom_runtime
        
        print(f"\nPerformance Results:")
        print(f"  Reference runtime: {ref_runtime:.3f} ms")
        print(f"  Custom runtime: {custom_runtime:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Show detailed stats
        if "runtime_stats" in exec_result:
            stats = exec_result["runtime_stats"]
            print(f"\nDetailed Statistics:")
            print(f"  Mean: {stats.mean:.3f} ms")
            print(f"  Std: {stats.std:.3f} ms")
            print(f"  Min: {stats.min:.3f} ms")
            print(f"  Max: {stats.max:.3f} ms")
            print(f"  Median: {stats.median:.3f} ms")
            print(f"  95th percentile: {stats.percentile_95:.3f} ms")
            print(f"  99th percentile: {stats.percentile_99:.3f} ms")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    client.close()


if __name__ == "__main__":
    main()