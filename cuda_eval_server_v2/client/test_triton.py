#!/usr/bin/env python3
"""
Test Triton kernel evaluation with the client library
"""

from kernel_eval_client import (
    KernelEvalClient,
    KernelCode,
    KernelType,
    IOContractBuilder,
    create_randn_spec
)


def test_triton_kernel():
    """Test Triton kernel evaluation"""
    print("Testing Triton Kernel Evaluation")
    print("-" * 40)
    
    client = KernelEvalClient("http://localhost:8000")
    
    # Triton kernel source
    triton_source = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
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
    
    # Build IOContract
    n_elements = 1024
    block_size = 256
    
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("x", create_randn_spec([n_elements], "float32", seed=42))
        .add_input_tensor("y", create_randn_spec([n_elements], "float32", seed=43))
        .add_output_tensor("output", [n_elements], "float32")
        .add_scalar("n_elements", "int", n_elements)
        .add_meta_param("BLOCK_SIZE", block_size)
        .set_grid(n_elements // block_size)
        .set_num_warps(4)
        .build()
    )
    
    # Create kernel
    triton_kernel = KernelCode(
        source_code=triton_source,
        kernel_type=KernelType.TRITON,
        io=io_contract
    )
    
    # Evaluate
    print("Evaluating Triton addition kernel...")
    result = client.evaluate(triton_kernel, num_trials=10)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        if exec_result["compiled"]:
            print(f"✓ Triton kernel compiled successfully")
            print(f"✓ Runtime: {exec_result['runtime']:.3f} ms")
            
            # Calculate bandwidth
            bytes_accessed = n_elements * 4 * 3  # 2 reads + 1 write
            bandwidth = (bytes_accessed / (1024**3)) / (exec_result['runtime'] / 1000)
            print(f"✓ Memory bandwidth: {bandwidth:.1f} GB/s")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
    else:
        print(f"✗ Evaluation failed: {result.get('error')}")
    
    client.close()
    print("\n✓ Triton test completed!")


if __name__ == "__main__":
    test_triton_kernel()