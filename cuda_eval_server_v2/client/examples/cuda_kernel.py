"""
CUDA kernel evaluation example with launch configuration
"""

from kernel_eval_client import (
    KernelEvalClient,
    KernelCode,
    KernelType,
    IOContractBuilder,
    create_randn_spec,
    create_zeros_spec,
    LaunchConfig,
    LaunchDim
)


def main():
    # Create client
    client = KernelEvalClient("http://localhost:8000")
    
    # Example 1: Simple vector addition
    print("Example 1: Vector Addition")
    print("-" * 40)
    
    cuda_source = """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
"""
    
    n = 1024 * 1024  # 1M elements
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    # Build IOContract with launch configuration
    io_contract = (
        IOContractBuilder()
        .add_input_tensor("a", create_randn_spec([n], "float32", seed=42))
        .add_input_tensor("b", create_randn_spec([n], "float32", seed=43))
        .add_output_tensor("c", [n], "float32")
        .add_scalar("n", "int", n)
        .set_grid(grid_size)
        .set_block(block_size)
        .build()
    )
    
    cuda_kernel = KernelCode(
        source_code=cuda_source,
        kernel_type=KernelType.CUDA,
        io=io_contract
    )
    
    # Evaluate kernel
    result = client.evaluate(cuda_kernel, num_trials=100)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        if exec_result["compiled"]:
            print(f"✓ CUDA kernel compiled successfully")
            print(f"  Runtime: {exec_result['runtime']:.3f} ms")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
    
    # Example 2: Matrix multiplication with shared memory
    print("\nExample 2: Matrix Multiplication")
    print("-" * 40)
    
    matmul_source = """
#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load tiles into shared memory
        if (row < M && k * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + k * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (k * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""
    
    # Matrix dimensions
    M, N, K = 512, 512, 512
    tile_size = 16
    
    # Calculate grid dimensions
    grid_x = (N + tile_size - 1) // tile_size
    grid_y = (M + tile_size - 1) // tile_size
    
    # Build IOContract with 2D launch config
    matmul_contract = (
        IOContractBuilder()
        .add_input_tensor("A", create_randn_spec([M, K], "float32", seed=100))
        .add_input_tensor("B", create_randn_spec([K, N], "float32", seed=101))
        .add_output_tensor("C", [M, N], "float32")
        .add_scalar("M", "int", M)
        .add_scalar("N", "int", N)
        .add_scalar("K", "int", K)
        .set_grid(grid_x, grid_y)
        .set_block(tile_size, tile_size)
        .build()
    )
    
    matmul_kernel = KernelCode(
        source_code=matmul_source,
        kernel_type=KernelType.CUDA,
        io=matmul_contract
    )
    
    # Evaluate matrix multiplication
    result = client.evaluate(matmul_kernel, num_trials=50)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        if exec_result["compiled"]:
            runtime = exec_result["runtime"]
            print(f"✓ Matrix multiplication compiled successfully")
            print(f"  Runtime: {runtime:.3f} ms")
            
            # Calculate GFLOPS
            ops = 2 * M * N * K  # Multiply-add operations
            gflops = (ops / 1e9) / (runtime / 1000)
            print(f"  Performance: {gflops:.1f} GFLOPS")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
    
    # Example 3: Multi-kernel file with explicit entrypoint
    print("\nExample 3: Multi-kernel with Entrypoint")
    print("-" * 40)
    
    multi_kernel_source = """
__global__ void kernel_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

__global__ void kernel_mul(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] * b[tid];
}

__global__ void kernel_scale(float* input, float* output, float scale, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) output[tid] = input[tid] * scale;
}
"""
    
    # Use kernel_scale as entrypoint
    scale_contract = (
        IOContractBuilder()
        .add_input_tensor("input", create_randn_spec([1024], "float32", seed=200))
        .add_output_tensor("output", [1024], "float32")
        .add_scalar("scale", "float", 2.5)
        .add_scalar("n", "int", 1024)
        .set_grid(4)
        .set_block(256)
        .build()
    )
    
    scale_kernel = KernelCode(
        source_code=multi_kernel_source,
        kernel_type=KernelType.CUDA,
        io=scale_contract,
        metadata={
            "kernel_name": "kernel_scale",  # Specify which kernel to use
            "compiler_options": ["--use_fast_math"]
        }
    )
    
    result = client.evaluate(scale_kernel, num_trials=100)
    
    if result["status"] == "success":
        exec_result = result["kernel_exec_result"]
        if exec_result["compiled"]:
            print(f"✓ Scale kernel compiled successfully")
            print(f"  Runtime: {exec_result['runtime']:.3f} ms")
        else:
            print(f"✗ Compilation failed: {exec_result.get('compilation_error')}")
    
    client.close()


if __name__ == "__main__":
    main()