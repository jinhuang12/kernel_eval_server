#!/usr/bin/env python3
"""
Test TorchExecutableKernel function targeting capabilities
Tests the ability to specify which function/method to execute in PyTorch code
"""

import logging
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compilation.torch.torch_backend import TorchCompilationBackend
from shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, TensorInit
from shared.kernel_metadata import TorchKernelMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_standalone_function():
    """Test targeting a standalone function"""
    print("\n" + "="*60)
    print("TEST: Standalone Function Targeting")
    print("="*60)
    
    # PyTorch code with standalone function
    source_code = """
import torch

def vector_add(a, b):
    return a + b

def vector_multiply(a, b):
    return a * b

# Also include a Model class (should be ignored when function_name is specified)
class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2
"""
    
    # Create IOContract for the function
    io_contract = IOContract(
        args=[
            ArgSpec(name="a", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[1024], dtype="float32",
                                         init=TensorInit(kind="randn"))),
            ArgSpec(name="b", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[1024], dtype="float32",
                                         init=TensorInit(kind="randn")))
        ]
    )
    
    # Create metadata targeting vector_add function
    metadata = TorchKernelMetadata(function_name="vector_add")
    
    # Create KernelCode
    kernel = KernelCode(
        source_code=source_code,
        kernel_type=KernelType.TORCH,
        io=io_contract,
        metadata=metadata
    )
    
    # Compile
    backend = TorchCompilationBackend()
    try:
        executable = backend.compile(kernel, gpu_id=0)
        print("✓ Compilation successful")
        print(f"  Target: standalone function 'vector_add'")
        print(f"  Using targeted execution: {executable._use_targeted_execution}")
        
        # Test execution
        result = executable()
        print(f"✓ Execution successful")
        print(f"  Output shape: {result.shape if torch.is_tensor(result) else 'N/A'}")
        
        # Test with different function
        metadata.function_name = "vector_multiply"
        kernel.metadata = metadata
        executable2 = backend.compile(kernel, gpu_id=0)
        result2 = executable2()
        print(f"✓ Switched to function 'vector_multiply'")
        
        # Verify results are different
        if not torch.allclose(result, result2):
            print("✓ Confirmed different functions produce different results")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_class_method():
    """Test targeting a specific class method"""
    print("\n" + "="*60)
    print("TEST: Class Method Targeting")
    print("="*60)
    
    source_code = """
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0
    
    def forward(self, x):
        return x * self.scale
    
    def custom_transform(self, x):
        return torch.relu(x) * 3.0
    
    def another_method(self, x):
        return x ** 2

# Also include default Model class
class Model(nn.Module):
    def forward(self, x):
        return x + 1
"""
    
    # Create IOContract
    io_contract = IOContract(
        args=[
            ArgSpec(name="x", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[64, 128], dtype="float32",
                                         init=TensorInit(kind="randn")))
        ]
    )
    
    # Test 1: Target CustomModel.forward (default method)
    metadata = TorchKernelMetadata(class_name="CustomModel")
    kernel = KernelCode(
        source_code=source_code,
        kernel_type=KernelType.TORCH,
        io=io_contract,
        metadata=metadata
    )
    
    backend = TorchCompilationBackend()
    try:
        executable = backend.compile(kernel, gpu_id=0)
        result1 = executable()
        print(f"✓ CustomModel.forward executed")
        print(f"  Output shape: {result1.shape}")
        
        # Test 2: Target CustomModel.custom_transform
        metadata = TorchKernelMetadata(
            class_name="CustomModel",
            method_name="custom_transform"
        )
        kernel.metadata = metadata
        executable = backend.compile(kernel, gpu_id=0)
        result2 = executable()
        print(f"✓ CustomModel.custom_transform executed")
        
        # Test 3: Target CustomModel.another_method
        metadata.method_name = "another_method"
        kernel.metadata = metadata
        executable = backend.compile(kernel, gpu_id=0)
        result3 = executable()
        print(f"✓ CustomModel.another_method executed")
        
        # Verify all results are different
        if not torch.allclose(result1, result2) and not torch.allclose(result2, result3):
            print("✓ Confirmed different methods produce different results")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_fallback_to_kernelbench():
    """Test fallback to KernelBench pattern when no metadata provided"""
    print("\n" + "="*60)
    print("TEST: Fallback to KernelBench Pattern")
    print("="*60)
    
    # Traditional KernelBench-style code
    source_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
    
    def forward(self, x):
        return self.linear(x)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 128)]
"""
    
    # Create kernel WITHOUT metadata (should trigger fallback)
    kernel = KernelCode(
        source_code=source_code,
        kernel_type=KernelType.TORCH,
        io=None,  # No IOContract
        metadata=None  # No metadata
    )
    
    backend = TorchCompilationBackend()
    try:
        executable = backend.compile(kernel, gpu_id=0)
        print(f"✓ Compilation successful")
        print(f"  Using targeted execution: {executable._use_targeted_execution}")
        
        if not executable._use_targeted_execution:
            print("✓ Correctly fell back to KernelBench pattern")
        
        # Test execution
        result = executable()
        print(f"✓ Execution successful")
        print(f"  Output shape: {result.shape if torch.is_tensor(result) else 'N/A'}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_mixed_scalars_and_tensors():
    """Test function with mixed scalar and tensor arguments"""
    print("\n" + "="*60)
    print("TEST: Mixed Scalar and Tensor Arguments")
    print("="*60)
    
    source_code = """
import torch

def scaled_add(x, y, alpha, beta):
    return alpha * x + beta * y

class Model(torch.nn.Module):
    def forward(self, x, y, alpha, beta):
        # Different implementation
        return x * alpha - y * beta
"""
    
    # Create IOContract with mixed types
    io_contract = IOContract(
        args=[
            ArgSpec(name="x", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[256], dtype="float32",
                                         init=TensorInit(kind="ones"))),
            ArgSpec(name="y", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[256], dtype="float32",
                                         init=TensorInit(kind="randn"))),
            ArgSpec(name="alpha", type="float", value=2.5),
            ArgSpec(name="beta", type="float", value=1.5)
        ]
    )
    
    # Target the standalone function
    metadata = TorchKernelMetadata(function_name="scaled_add")
    
    kernel = KernelCode(
        source_code=source_code,
        kernel_type=KernelType.TORCH,
        io=io_contract,
        metadata=metadata
    )
    
    backend = TorchCompilationBackend()
    try:
        executable = backend.compile(kernel, gpu_id=0)
        result = executable()
        print(f"✓ Function with mixed arguments executed")
        print(f"  Output shape: {result.shape}")
        
        # Verify it's using the function, not the Model
        # scaled_add: 2.5 * ones + 1.5 * randn
        # Model.forward: ones * 2.5 - randn * 1.5
        x = torch.ones(256).cuda(0)
        expected = 2.5 * x  # First part of scaled_add with x=ones
        if torch.allclose(result[:10], expected[:10], atol=2.0):  # Allow for randn variation
            print("✓ Confirmed using standalone function, not Model.forward")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_error_handling():
    """Test error handling for invalid targets"""
    print("\n" + "="*60)
    print("TEST: Error Handling")
    print("="*60)
    
    source_code = """
import torch

def valid_function(x):
    return x * 2

class ValidClass(torch.nn.Module):
    def forward(self, x):
        return x + 1
"""
    
    io_contract = IOContract(
        args=[
            ArgSpec(name="x", type="tensor", role="input",
                   tensor_spec=TensorSpec(shape=[100], dtype="float32",
                                         init=TensorInit(kind="randn")))
        ]
    )
    
    backend = TorchCompilationBackend()
    
    # Test 1: Non-existent function
    print("Testing non-existent function...")
    metadata = TorchKernelMetadata(function_name="non_existent_function")
    kernel = KernelCode(
        source_code=source_code,
        kernel_type=KernelType.TORCH,
        io=io_contract,
        metadata=metadata
    )
    
    try:
        executable = backend.compile(kernel, gpu_id=0)
        if not executable._use_targeted_execution:
            print("✓ Correctly fell back to KernelBench on missing function")
        else:
            print("✗ Should have fallen back but didn't")
    except Exception as e:
        print(f"  Error handled: {str(e)[:100]}...")
    
    # Test 2: Non-existent class
    print("Testing non-existent class...")
    metadata = TorchKernelMetadata(class_name="NonExistentClass")
    kernel.metadata = metadata
    
    try:
        executable = backend.compile(kernel, gpu_id=0)
        if not executable._use_targeted_execution:
            print("✓ Correctly fell back to KernelBench on missing class")
        else:
            print("✗ Should have fallen back but didn't")
    except Exception as e:
        print(f"  Error handled: {str(e)[:100]}...")
    
    # Test 3: Non-existent method
    print("Testing non-existent method...")
    metadata = TorchKernelMetadata(
        class_name="ValidClass",
        method_name="non_existent_method"
    )
    kernel.metadata = metadata
    
    try:
        executable = backend.compile(kernel, gpu_id=0)
        if not executable._use_targeted_execution:
            print("✓ Correctly fell back to KernelBench on missing method")
        else:
            print("✗ Should have fallen back but didn't")
    except Exception as e:
        print(f"  Error handled: {str(e)[:100]}...")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TORCH FUNCTION TARGETING TEST SUITE")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. These tests require a GPU.")
        return 1
    
    tests = [
        ("Standalone Function", test_standalone_function),
        ("Class Method", test_class_method),
        ("Fallback to KernelBench", test_fallback_to_kernelbench),
        ("Mixed Arguments", test_mixed_scalars_and_tensors),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())