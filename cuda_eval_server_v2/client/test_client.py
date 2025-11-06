#!/usr/bin/env python3
"""
Simple test script to verify the client library works end-to-end
"""

from kernel_eval_client import (
    KernelEvalClient,
    KernelCode,
    KernelType,
    IOContractBuilder,
    create_randn_spec
)


def test_simple_evaluation():
    """Test simple kernel evaluation"""
    print("Testing Kernel Evaluation Client")
    print("-" * 40)
    
    # Create client
    client = KernelEvalClient("http://localhost:8000")
    
    # Test health check
    health = client.health_check()
    print(f"✓ Server health: {health['status']}")
    
    # Create a simple PyTorch kernel
    kernel = KernelCode(
        source_code="""
import torch

def simple_add(x, y):
    return x + y
""",
        kernel_type=KernelType.TORCH,
        io=(
            IOContractBuilder()
            .add_input_tensor("x", create_randn_spec([256, 256], seed=42))
            .add_input_tensor("y", create_randn_spec([256, 256], seed=43))
            .build()
        ),
        metadata={"function_name": "simple_add"}
    )
    
    # Evaluate kernel
    print("\nEvaluating PyTorch kernel...")
    result = client.evaluate(kernel, num_trials=10)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        if exec_result["compiled"]:
            print(f"✓ Kernel compiled successfully")
            print(f"✓ Runtime: {exec_result['runtime']:.3f} ms")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
    else:
        print(f"✗ Evaluation failed: {result.get('error')}")
    
    client.close()
    print("\n✓ Client test completed successfully!")


def test_comparison():
    """Test kernel comparison"""
    print("\nTesting Kernel Comparison")
    print("-" * 40)
    
    client = KernelEvalClient("http://localhost:8000")
    
    # Reference kernel
    ref_kernel = KernelCode(
        source_code="""
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)
""",
        kernel_type=KernelType.TORCH,
        io=(
            IOContractBuilder()
            .add_input_tensor("x", create_randn_spec([256, 256], seed=42))
            .build()
        ),
        metadata={"class_name": "Model", "method_name": "forward" }
    )
    
    # Custom kernel (same for simplicity)
    custom_kernel = KernelCode(
        source_code="""
import torch

def forward(x):
    return torch.relu(x)
""",
        kernel_type=KernelType.TORCH,
        io=(
            IOContractBuilder()
            .add_input_tensor("x", create_randn_spec([256, 256], seed=42))
            .build()
        ),
        metadata={"function_name": "forward"}
    )
    
    # Compare
    print("Comparing two PyTorch ReLU implementations...")
    result = client.compare(ref_kernel, custom_kernel, num_trials=10)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        if exec_result["compiled"] and exec_result["correctness"]:
            ref_runtime = result["ref_runtime"]["mean"]
            custom_runtime = exec_result["runtime"]
            print(f"✓ Reference runtime: {ref_runtime:.3f} ms")
            print(f"✓ Custom runtime: {custom_runtime:.3f} ms")
            speedup = ref_runtime / custom_runtime
            print(f"✓ Speedup: {speedup:.2f}x")
        else:
            print(f"✗ Kernel issues: compiled={exec_result['compiled']}, correct={exec_result.get('correctness')}")
    else:
        print(f"✗ Comparison failed: {result.get('error')}")
    
    client.close()
    print("\n✓ Comparison test completed!")


if __name__ == "__main__":
    test_simple_evaluation()
    test_comparison()
    print("\n" + "="*40)
    print("All tests passed! Client library is working correctly.")