#!/usr/bin/env python3
"""
Test script to debug CuPy kernel execution with PyTorch stream synchronization
"""

import torch
import cupy
import sys
import traceback
from shared.models import IOContract, ArgSpec, TensorSpec, TensorInit, LaunchConfig, LaunchDim
from io_contract import IOContractManager

def test_cuda_kernel_execution():
    """Test CUDA kernel execution with stream synchronization"""
    
    # Simple vector add CUDA kernel
    cuda_source = '''
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
'''
    
    print("1. Setting up CUDA device...")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"   Device: {device}")
    
    print("\n2. Creating IOContract for vector addition...")
    n = 1024
    io_contract = IOContract(
        args=[
            ArgSpec(
                name="a",
                type="tensor",
                role="input",
                tensor_spec=TensorSpec(
                    dtype="float32",
                    shape=[n],
                    init=TensorInit(kind="full", fill_value=2.0)
                )
            ),
            ArgSpec(
                name="b",
                type="tensor",
                role="input", 
                tensor_spec=TensorSpec(
                    dtype="float32",
                    shape=[n],
                    init=TensorInit(kind="full", fill_value=3.0)
                )
            ),
            ArgSpec(
                name="c",
                type="tensor",
                role="output",
                tensor_spec=TensorSpec(
                    dtype="float32",
                    shape=[n],
                    init=TensorInit(kind="zeros")
                )
            ),
            ArgSpec(
                name="n",
                type="int",
                role="input",
                value=n
            )
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=(n + 256 - 1) // 256, y=1, z=1),
            block=LaunchDim(x=256, y=1, z=1)
        )
    )
    
    io_manager = IOContractManager()
    print("\n3. Generating tensors from IOContract...")
    inputs = io_manager.generate_inputs(io_contract, device)
    
    a, b, c, n_val = inputs
    print(f"   Created tensors of size {n}")
    
    print("\n4. Getting PyTorch CUDA stream...")
    torch_stream = torch.cuda.current_stream(device)
    print(f"   PyTorch stream: {torch_stream}")
    print(f"   Stream cuda_stream: {torch_stream.cuda_stream}")
    
    print("\n5. Compiling CuPy kernel...")
    try:
        with cupy.cuda.Device(device.index):
            cupy_module = cupy.RawModule(code=cuda_source)
            cupy_kernel = cupy_module.get_function("vector_add")
            print("   Kernel compiled successfully")
    except Exception as e:
        print(f"   ERROR compiling kernel: {e}")
        traceback.print_exc()
        return False
    
    print("\n6. Converting tensors to CuPy arrays with stream context...")
    try:
        with cupy.cuda.Device(device.index):
            # Use PyTorch's CUDA stream for CuPy operations
            with cupy.cuda.ExternalStream(torch_stream.cuda_stream):
                print("   Created ExternalStream successfully")
                
                # Ensure tensors are contiguous
                if not a.is_contiguous():
                    a = a.contiguous()
                if not b.is_contiguous():
                    b = b.contiguous()
                if not c.is_contiguous():
                    c = c.contiguous()
                
                # Convert to CuPy arrays
                a_cupy = cupy.asarray(a)
                b_cupy = cupy.asarray(b)
                c_cupy = cupy.asarray(c)
                print(f"   Converted tensors to CuPy arrays")
                print(f"   a_cupy shape: {a_cupy.shape}, dtype: {a_cupy.dtype}")
                
                print("\n7. Executing kernel...")
                # Launch configuration from IOContract
                grid = io_contract.launch.grid
                block = io_contract.launch.block
                
                # Execute kernel
                cupy_kernel(
                    (grid.x, grid.y, grid.z),  # grid
                    (block.x, block.y, block.z),  # block
                    (a_cupy, b_cupy, c_cupy, n_val)  # arguments
                )
                print(f"   Kernel launched with grid=({grid.x},{grid.y},{grid.z}), block=({block.x},{block.y},{block.z})")
                
                print("\n8. Synchronizing stream...")
                torch_stream.synchronize()
                print("   Stream synchronized")
                
                print("\n9. Verifying results...")
                # The c tensor should now have a + b
                expected = a + b
                diff = torch.abs(c - expected).max().item()
                print(f"   Max difference: {diff}")
                
                if diff < 1e-5:
                    print("   ✅ Results match!")
                    return True
                else:
                    print(f"   ❌ Results don't match (diff={diff})")
                    return False
                    
    except Exception as e:
        print(f"\n   ERROR during execution: {e}")
        traceback.print_exc()
        return False

def test_without_ncu():
    """Test without NCU profiling"""
    print("="*60)
    print("Testing CUDA kernel execution WITHOUT NCU")
    print("="*60)
    
    success = test_cuda_kernel_execution()
    
    if success:
        print("\n✅ Test PASSED without NCU")
    else:
        print("\n❌ Test FAILED without NCU")
    
    return success

def test_with_ncu():
    """Test with NCU profiling simulation"""
    print("\n" + "="*60)
    print("Testing CUDA kernel execution WITH NCU environment")
    print("="*60)
    
    # Simulate NCU environment variables
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    print("Set CUDA_LAUNCH_BLOCKING=1 and TORCH_USE_CUDA_DSA=1")
    
    success = test_cuda_kernel_execution()
    
    if success:
        print("\n✅ Test PASSED with NCU environment")
    else:
        print("\n❌ Test FAILED with NCU environment")
    
    return success

if __name__ == "__main__":
    print("CUDA Stream Synchronization Test")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CuPy version: {cupy.__version__}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print()
    
    # Test without NCU
    success1 = test_without_ncu()
    
    # Test with NCU
    success2 = test_with_ncu()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if success1 and success2:
        print("✅ All tests PASSED")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED")
        if not success1:
            print("  - Failed without NCU")
        if not success2:
            print("  - Failed with NCU environment")
        sys.exit(1)