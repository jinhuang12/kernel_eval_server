"""
Unit tests for compilation backends
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compilation.torch.torch_backend import TorchCompilationBackend
from compilation.compiler_service import CompilationService
from shared.models import KernelCode, KernelType, IOContract, ArgSpec, TensorSpec, TensorInit, CompilationRequest
from shared.executable_kernels import TorchExecutableKernel
from tests.fixtures.kernels import KernelLibrary

# Try to import optional dependencies
try:
    from compilation.triton.triton_backend import TritonCompilationBackend
    from shared.executable_kernels import TritonExecutableKernel
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    TritonExecutableKernel = None

try:
    from compilation.cuda.cuda_backend import CudaCompilationBackend
    from shared.executable_kernels import CudaExecutableKernel
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    CudaExecutableKernel = None


@pytest.mark.unit
class TestTorchBackend:
    """Tests for Torch compilation backend"""
    
    def test_compile_simple_kernel(self):
        """Test compiling a simple torch kernel"""
        backend = TorchCompilationBackend()
        kernel_data = KernelLibrary.torch_add()
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TORCH.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io"),
            "metadata": kernel_data.get("metadata")
        })
        
        result = backend.compile(kernel_code, gpu_id=0)
        
        # Backend returns executable kernel directly
        assert result is not None
        assert isinstance(result, TorchExecutableKernel)
    
    def test_compile_with_imports(self):
        """Test kernel with multiple imports"""
        backend = TorchCompilationBackend()
        kernel_data = KernelLibrary.torch_gelu()
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TORCH.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io"),
            "metadata": kernel_data.get("metadata")
        })
        
        result = backend.compile(kernel_code, gpu_id=0)
        
        # Backend returns executable kernel directly
        assert result is not None
        assert isinstance(result, TorchExecutableKernel)
    
    def test_compile_syntax_error(self):
        """Test handling of syntax errors"""
        backend = TorchCompilationBackend()
        kernel_data = KernelLibrary.compilation_error()
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TORCH.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io"),
            "metadata": kernel_data.get("metadata")
        })
        
        # Compilation error kernels should raise an exception
        with pytest.raises(Exception) as exc_info:
            result = backend.compile(kernel_code, gpu_id=0)

        # Check that it raised an exception (any exception indicates compilation failure)
        assert exc_info.value is not None
    
    def test_execute_kernel(self):
        """Test executing a compiled kernel"""
        backend = TorchCompilationBackend()
        kernel_data = KernelLibrary.torch_add()
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TORCH.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io"),
            "metadata": kernel_data.get("metadata")
        })
        
        kernel = backend.compile(kernel_code, gpu_id=0)
        assert kernel is not None
        
        # Create test inputs
        x = torch.randn(100)
        y = torch.randn(100)
        
        # Execute kernel
        output = kernel(x, y)
        
        # Verify output
        expected = torch.add(x, y)
        assert torch.allclose(output, expected)


@pytest.mark.unit  
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestTritonBackend:
    """Tests for Triton compilation backend"""
    
    def test_compile_triton_kernel(self):
        """Test compiling a Triton kernel"""
        backend = TritonCompilationBackend()
        kernel_data = KernelLibrary.triton_add()
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TRITON.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io") or kernel_data.get("io_contract"),
            "metadata": kernel_data.get("metadata")
        })
        
        result = backend.compile(kernel_code, gpu_id=0)
        
        # Backend returns executable kernel directly
        assert result is not None
        assert isinstance(result, TritonExecutableKernel)
    
    def test_triton_without_iocontract(self):
        """Test that Triton requires IOContract"""
        backend = TritonCompilationBackend()
        
        kernel_code = KernelCode(
            kernel_type=KernelType.TRITON.value,
            source_code="pass",
            io=None
        )
        
        # Should raise error about missing IOContract
        with pytest.raises(ValueError) as exc_info:
            result = backend.compile(kernel_code, gpu_id=0)
        
        assert "io" in str(exc_info.value).lower()
    
    def test_execute_triton_kernel(self):
        """Test executing a compiled Triton kernel"""
        backend = TritonCompilationBackend()
        kernel_data = KernelLibrary.triton_add()

        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TRITON.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io") or kernel_data.get("io_contract"),
            "metadata": kernel_data.get("metadata")
        })

        kernel = backend.compile(kernel_code, gpu_id=0)
        assert kernel is not None

        # Create test inputs
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")

        # Execute kernel
        output = kernel(x, y)

        # Verify output
        expected = torch.add(x, y)
        assert torch.allclose(output, expected, rtol=1e-5)

    def test_triton_batched_matmul(self):
        """Test complex Triton kernel with batched matrix multiplication"""
        from validation.correctness_validator import ExecutableValidator

        backend = TritonCompilationBackend()

        # Batched matmul kernel source
        src = """
import torch
import triton
import triton.language as tl

@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, M, N, K,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs for batch, M, and N dimensions
    pid_batch = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Compute offsets for current batch
    batch_offset_a = pid_batch * stride_a_batch
    batch_offset_b = pid_batch * stride_b_batch
    batch_offset_c = pid_batch * stride_c_batch

    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main computation loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block
        a_ptrs = a_ptr + batch_offset_a + (offs_m[:, None] * stride_a_m + (k + offs_k[None, :]) * stride_a_k)
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B block
        b_ptrs = b_ptr + batch_offset_b + ((k + offs_k[:, None]) * stride_b_k + offs_n[None, :] * stride_b_n)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate using dot product
        accumulator += tl.dot(a, b)

    # Store result
    c_ptrs = c_ptr + batch_offset_c + (offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
"""

        # Create IOContract for batched matmul
        from shared.models import LaunchConfig, LaunchDim
        io = IOContract.from_dict({
            "args": [
                {
                    "name": "a_ptr",
                    "type": "tensor",
                    "role": "input",
                    "tensor_spec": {
                        "shape": [128, 128, 256],
                        "dtype": "float32",
                        "init": {"kind": "randn", "seed": 42}
                    }
                },
                {
                    "name": "b_ptr",
                    "type": "tensor",
                    "role": "input",
                    "tensor_spec": {
                        "shape": [128, 256, 512],
                        "dtype": "float32",
                        "init": {"kind": "randn", "seed": 43}
                    }
                },
                {
                    "name": "c_ptr",
                    "type": "tensor",
                    "role": "output",
                    "tensor_spec": {
                        "shape": [128, 128, 512],
                        "dtype": "float32"
                    }
                },
                {"name": "batch_size", "type": "int", "value": 128},
                {"name": "M", "type": "int", "value": 128},
                {"name": "N", "type": "int", "value": 512},
                {"name": "K", "type": "int", "value": 256},
                {"name": "stride_a_batch", "type": "int", "value": 32768},
                {"name": "stride_a_m", "type": "int", "value": 256},
                {"name": "stride_a_k", "type": "int", "value": 1},
                {"name": "stride_b_batch", "type": "int", "value": 131072},
                {"name": "stride_b_k", "type": "int", "value": 512},
                {"name": "stride_b_n", "type": "int", "value": 1},
                {"name": "stride_c_batch", "type": "int", "value": 65536},
                {"name": "stride_c_m", "type": "int", "value": 512},
                {"name": "stride_c_n", "type": "int", "value": 1},
                {"name": "BLOCK_SIZE_M", "type": "int", "value": 64, "is_meta": True},
                {"name": "BLOCK_SIZE_N", "type": "int", "value": 64, "is_meta": True},
                {"name": "BLOCK_SIZE_K", "type": "int", "value": 32, "is_meta": True}
            ],
            "launch": {
                "grid": {"x": 128, "y": 2, "z": 8},
                "num_warps": 4,
                "num_stages": 3
            }
        })

        kernel_code = KernelCode(
            source_code=src,
            kernel_type=KernelType.TRITON,
            io=io
        )

        # Compile kernel
        executable = backend.compile(kernel_code, gpu_id=0)
        assert executable is not None

        # Validate execution
        validator = ExecutableValidator()
        result = validator.validate(executable, torch.device('cuda:0'), 2, 'test_job')

        assert result is not None
        assert result["is_valid"] == True


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA backend not available")  
class TestCudaBackend:
    """Tests for CUDA compilation backend"""
    
    def test_compile_cuda_kernel(self):
        """Test compiling a CUDA kernel"""
        backend = CudaCompilationBackend()
        
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=[1024], dtype="float32"), role="input"),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(shape=[1024], dtype="float32"), role="input"),
                ArgSpec(name="out", type="tensor", tensor_spec=TensorSpec(shape=[1024], dtype="float32"), role="output")
            ]
        )
        
        kernel_code = KernelCode(
            kernel_type=KernelType.CUDA.value,
            source_code="""
__global__ void vector_add(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}
""",
            io=io_contract
        )
        
        result = backend.compile(kernel_code, gpu_id=0)
        
        # Backend returns executable kernel directly
        assert result is not None
        assert isinstance(result, CudaExecutableKernel)


@pytest.mark.unit
class TestCompilerService:
    """Tests for the compiler service orchestrator"""
    
    def test_get_backend_for_kernel_type(self):
        """Test backend selection based on kernel type"""
        service = CompilationService()
        
        # Torch backend
        backend = service.backends[KernelType.TORCH]
        assert isinstance(backend, TorchCompilationBackend)
        
        # Triton backend (if available)
        if HAS_TRITON:
            backend = service.backends[KernelType.TRITON]
            assert isinstance(backend, TritonCompilationBackend)
    
    def test_compile_via_service(self):
        """Test compilation through the service"""
        service = CompilationService()
        kernel_data = KernelLibrary.torch_add()
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": KernelType.TORCH.value,
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io") or kernel_data.get("io_contract"),
            "metadata": kernel_data.get("metadata")
        })
        
        # Create compilation request
        request = CompilationRequest(kernel_code=kernel_code)
        
        result = service.compile_kernel(request, gpu_id=0)
        
        # Service returns CompilationResult
        assert result.compiles == True
        assert result.kernel is not None
        assert result.error is None
    
    @pytest.mark.parametrize("kernel_name", [
        "torch_add",
        "torch_matmul", 
        "torch_gelu"
    ])
    def test_compile_multiple_kernels(self, kernel_name):
        """Test compiling different kernel types"""
        service = CompilationService()
        kernel_data = KernelLibrary.get_kernel(kernel_name)
        
        # Create KernelCode from dict to properly handle IOContract conversion
        kernel_code = KernelCode.from_dict({
            "kernel_type": kernel_data["kernel_type"],
            "source_code": kernel_data["source_code"],
            "io": kernel_data.get("io") or kernel_data.get("io_contract"),
            "metadata": kernel_data.get("metadata")
        })
        
        # Create compilation request
        request = CompilationRequest(kernel_code=kernel_code)
        
        result = service.compile_kernel(request, gpu_id=0)
        
        # Service returns CompilationResult
        assert result.compiles == True
        assert result.kernel is not None