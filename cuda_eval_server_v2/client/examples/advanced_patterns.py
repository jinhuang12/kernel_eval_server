"""
Advanced patterns for using the Kernel Evaluation Client
"""

import base64
import numpy as np
from kernel_eval_client import (
    KernelEvalClient,
    KernelCode,
    KernelType,
    IOContractBuilder,
    TensorSpec,
    TensorData,
    TensorInit,
    create_randn_spec,
    to_json,
    save_to_file,
    load_from_file
)


def example_custom_tensor_data():
    """Example: Send custom tensor data instead of server generation"""
    print("Example: Custom Tensor Data")
    print("-" * 40)
    
    # Create custom numpy array
    custom_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    
    # Encode as base64
    data_bytes = custom_data.tobytes()
    data_b64 = base64.b64encode(data_bytes).decode('utf-8')
    
    # Create TensorSpec with literal data
    tensor_spec = TensorSpec(
        shape=list(custom_data.shape),
        dtype="float32",
        data=TensorData(
            data_b64=data_b64,
            dtype="float32",
            shape=list(custom_data.shape),
            compress="none"
        )
    )
    
    # Build IOContract with custom data
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("custom_input", tensor_spec)
        .add_output_tensor("output", list(custom_data.shape), "float32")
        .build()
    )
    
    print(f"Custom tensor shape: {custom_data.shape}")
    print(f"Custom tensor data:\n{custom_data}")
    
    return io_contract


def example_mixed_precision():
    """Example: Mixed precision kernels with different dtypes"""
    print("\nExample: Mixed Precision")
    print("-" * 40)
    
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("x_fp32", create_randn_spec([1024], "float32"))
        .add_input_tensor("x_fp16", create_randn_spec([1024], "float16"))
        .add_input_tensor("x_int32", create_randn_spec([1024], "int32"))
        .add_output_tensor("out_fp32", [1024], "float32")
        .add_output_tensor("out_fp16", [1024], "float16")
        .build()
    )
    
    print("Created mixed precision IOContract with:")
    print("  - float32 input/output")
    print("  - float16 input/output")
    print("  - int32 input")
    
    return io_contract


def example_save_load_contract():
    """Example: Save and load IOContract to/from JSON files"""
    print("\nExample: Save/Load IOContract")
    print("-" * 40)
    
    # Create an IOContract
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("x", create_randn_spec([512, 512], seed=42))
        .add_input_tensor("y", create_randn_spec([512, 512], seed=43))
        .add_output_tensor("z", [512, 512])
        .set_grid(16, 16)
        .set_block(16, 16)
        .build()
    )
    
    # Save to file
    filename = "example_contract.json"
    save_to_file(io_contract, filename)
    print(f"Saved IOContract to {filename}")
    
    # Load from file
    loaded_contract = load_from_file(filename, "IOContract")
    print(f"Loaded IOContract from {filename}")
    print(f"  Number of args: {len(loaded_contract.args)}")
    print(f"  Grid dimensions: {loaded_contract.launch.grid.x}x{loaded_contract.launch.grid.y}")
    
    # Convert to JSON string
    json_str = to_json(io_contract, pretty=True)
    print(f"\nJSON representation (first 200 chars):")
    print(json_str[:200] + "...")
    
    return loaded_contract


def example_pytorch_function_targeting():
    """Example: Target specific functions in PyTorch code"""
    print("\nExample: PyTorch Function Targeting")
    print("-" * 40)
    
    source_code = """
import torch
import torch.nn as nn

def my_custom_function(x, y, alpha=2.0):
    '''Custom function to be targeted'''
    return x + alpha * y

def another_function(x):
    '''This won't be called'''
    return x * 2

class MyModel(nn.Module):
    def forward(self, x, y):
        '''This won't be called either'''
        return x - y
"""
    
    # Create IOContract for the function
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("x", create_randn_spec([256, 256]))
        .add_input_tensor("y", create_randn_spec([256, 256]))
        .add_scalar("alpha", "float", 3.0)
        .build()
    )
    
    # Create kernel targeting specific function
    kernel = KernelCode(
        source_code=source_code,
        kernel_type=KernelType.TORCH,
        io=io_contract,
        metadata={
            "function_name": "my_custom_function"  # Target this function
        }
    )
    
    print("Created kernel targeting function: my_custom_function")
    print("  With arguments: x, y, alpha=3.0")
    
    return kernel


def example_batch_evaluation():
    """Example: Evaluate multiple configurations"""
    print("\nExample: Batch Evaluation")
    print("-" * 40)
    
    client = KernelEvalClient("http://localhost:8000")
    
    # Test different input sizes
    sizes = [256, 512, 1024, 2048]
    results = []
    
    for size in sizes:
        io_contract = (
            IOContractBuilder()
            .add_input_tensor("x", create_randn_spec([size, size]))
            .add_output_tensor("y", [size, size])
            .build()
        )
        
        kernel = KernelCode(
            source_code="import torch\ndef process(x): return torch.relu(x)",
            kernel_type=KernelType.TORCH,
            io=io_contract,
            metadata={"function_name": "process"}
        )
        
        try:
            result = client.evaluate(kernel, num_trials=10)
            if result["status"] == "success":
                runtime = result["kernel_exec_result"]["runtime"]
                results.append((size, runtime))
                print(f"  Size {size}x{size}: {runtime:.3f} ms")
        except Exception as e:
            print(f"  Size {size}x{size}: Failed - {e}")
    
    client.close()
    
    # Show scaling
    if len(results) > 1:
        print("\nScaling Analysis:")
        base_size, base_time = results[0]
        for size, runtime in results[1:]:
            scale_factor = (size / base_size) ** 2  # Quadratic for matrix ops
            time_factor = runtime / base_time
            efficiency = scale_factor / time_factor
            print(f"  {size}x{size}: {efficiency:.2f}x efficiency vs {base_size}x{base_size}")
    
    return results


def example_custom_init_patterns():
    """Example: Various tensor initialization patterns"""
    print("\nExample: Custom Initialization Patterns")
    print("-" * 40)
    
    # Different initialization patterns
    patterns = [
        ("Random Normal (mean=5, std=2)", 
         TensorInit(kind="randn", seed=42, mean=5.0, std=2.0)),
        
        ("Uniform [-1, 1]",
         TensorInit(kind="uniform", seed=43, low=-1.0, high=1.0)),
        
        ("All Zeros",
         TensorInit(kind="zeros")),
        
        ("All Ones",
         TensorInit(kind="ones")),
        
        ("Constant (3.14)",
         TensorInit(kind="full", fill_value=3.14)),
        
        ("Sequential (0, 0.1, 0.2, ...)",
         TensorInit(kind="arange", start=0.0, step=0.1))
    ]
    
    for name, init in patterns:
        tensor_spec = TensorSpec(
            shape=[10],  # Small for display
            dtype="float32",
            init=init
        )
        print(f"  {name}")
        print(f"    Spec: {init.to_dict()}")
    
    return patterns


def main():
    """Run all examples"""
    print("=" * 50)
    print("Advanced Kernel Evaluation Client Patterns")
    print("=" * 50)
    
    # Run examples
    custom_io = example_custom_tensor_data()
    mixed_io = example_mixed_precision()
    loaded_io = example_save_load_contract()
    pytorch_kernel = example_pytorch_function_targeting()
    init_patterns = example_custom_init_patterns()
    
    # Uncomment to run batch evaluation (requires server)
    # batch_results = example_batch_evaluation()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()