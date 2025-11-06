"""
Triton kernel evaluation example with IOContract
"""

from kernel_eval_client import (
    KernelEvalClient,
    KernelCode,
    KernelType,
    IOContractBuilder,
    create_randn_spec,
    create_ones_spec
)


def main():
    # Create client
    client = KernelEvalClient("http://localhost:8000")
    
    # Define Triton kernel for element-wise addition
    triton_source = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary check
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform computation
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)
"""
    
    # Build IOContract for Triton kernel
    n_elements = 1024 * 1024  # 1M elements
    block_size = 1024
    
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("x", create_randn_spec([n_elements], "float32", seed=42))
        .add_input_tensor("y", create_randn_spec([n_elements], "float32", seed=43))
        .add_output_tensor("output", [n_elements], "float32")
        .add_scalar("n_elements", "int", n_elements)
        .add_meta_param("BLOCK_SIZE", block_size)  # Triton constexpr
        .set_grid(n_elements // block_size)  # Grid size
        .set_num_warps(4)  # Triton-specific
        .build()
    )
    
    # Create kernel with IOContract
    triton_kernel = KernelCode(
        source_code=triton_source,
        kernel_type=KernelType.TRITON,
        io=io_contract
    )
    
    # Evaluate single kernel
    print("Evaluating Triton kernel...")
    result = client.evaluate(triton_kernel, num_trials=100)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        
        # Check compilation
        if exec_result["compiled"]:
            print("✓ Triton kernel compiled successfully")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
            return
        
        # Show performance
        runtime = exec_result["runtime"]
        print(f"\nPerformance:")
        print(f"  Runtime: {runtime:.3f} ms")
        
        # Calculate throughput
        bytes_accessed = n_elements * 4 * 3  # 2 reads + 1 write, 4 bytes per float
        bandwidth = (bytes_accessed / (1024**3)) / (runtime / 1000)  # GB/s
        print(f"  Memory bandwidth: {bandwidth:.1f} GB/s")
        
        # Show device metrics if available
        if "metadata" in exec_result and exec_result["metadata"]:
            metadata = exec_result["metadata"]
            if "device_metrics" in metadata and metadata["device_metrics"]:
                metrics = metadata["device_metrics"]
                if "speed_of_light" in metrics:
                    sol = metrics["speed_of_light"]
                    print(f"\nDevice Metrics (Speed of Light):")
                    print(f"  Compute throughput: {sol.get('compute_throughput_pct', 'N/A')}%")
                    print(f"  Memory throughput: {sol.get('memory_throughput_pct', 'N/A')}%")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Example 2: Compare with PyTorch reference
    print("\n" + "="*50)
    print("Comparing with PyTorch reference...")
    
    # PyTorch reference
    ref_kernel = KernelCode(
        source_code="""
import torch

def optimized_add(x, y):
    return x + y
""",
        kernel_type=KernelType.TORCH,
        io=io_contract,  # Use same IOContract
        metadata={
            "function_name": "optimized_add"  # Target specific function
        }
    )
    
    # Compare
    result = client.compare(ref_kernel, triton_kernel, num_trials=100)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        
        if exec_result["compiled"] and exec_result["correctness"]:
            ref_runtime = result["ref_runtime"]["mean"]
            triton_runtime = exec_result["runtime"]
            speedup = ref_runtime / triton_runtime
            
            print(f"\nComparison Results:")
            print(f"  PyTorch runtime: {ref_runtime:.3f} ms")
            print(f"  Triton runtime: {triton_runtime:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
    
    client.close()


if __name__ == "__main__":
    main()