"""
Test fixtures for MCP integration tests

This module provides kernel examples and IOContract definitions for testing
the MCP server with different kernel types.
"""

# ============================================================================
# TORCH KERNEL FIXTURES
# ============================================================================

SIMPLE_TORCH_KERNEL = """import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(1024)]
"""

TORCH_MATMUL_KERNEL = """import torch

class MatMul(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)

def get_inputs():
    return [torch.randn(32, 32), torch.randn(32, 32)]
"""

# ============================================================================
# TRITON KERNEL FIXTURES
# ============================================================================

SIMPLE_TRITON_ADD_KERNEL = """import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
"""

SIMPLE_TRITON_ADD_IO_CONTRACT = {
    "args": [
        {
            "name": "x_ptr",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "randn", "seed": 42}
            }
        },
        {
            "name": "y_ptr",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "ones"}
            }
        },
        {
            "name": "out_ptr",
            "type": "tensor",
            "role": "output",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32"
            }
        },
        {
            "name": "n",
            "type": "int",
            "value": 1024,
            "role": "input"
        },
        {
            "name": "BLOCK_SIZE",
            "type": "int",
            "value": 256,
            "role": "input",
            "is_meta": True
        }
    ],
    "launch": {
        "grid": {"x": 4, "y": 1, "z": 1},
        "num_warps": 4
    }
}

TRITON_VECTOR_SCALE_KERNEL = """import triton
import triton.language as tl

@triton.jit
def vector_scale_kernel(x_ptr, out_ptr, scale, n, BLOCK_SIZE: tl.constexpr):
    \"\"\"Scale a vector by a scalar\"\"\"
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * scale
    tl.store(out_ptr + offsets, output, mask=mask)
"""

TRITON_VECTOR_SCALE_IO_CONTRACT = {
    "args": [
        {
            "name": "x_ptr",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [2048],
                "dtype": "float32",
                "init": {"kind": "uniform", "low": 0.0, "high": 1.0, "seed": 123}
            }
        },
        {
            "name": "out_ptr",
            "type": "tensor",
            "role": "output",
            "tensor_spec": {
                "shape": [2048],
                "dtype": "float32"
            }
        },
        {
            "name": "scale",
            "type": "float",
            "value": 2.5,
            "role": "input"
        },
        {
            "name": "n",
            "type": "int",
            "value": 2048,
            "role": "input"
        },
        {
            "name": "BLOCK_SIZE",
            "type": "int",
            "value": 512,
            "role": "input",
            "is_meta": True
        }
    ],
    "launch": {
        "grid": {"x": 4},
        "num_warps": 8
    }
}

# ============================================================================
# CUDA KERNEL FIXTURES
# ============================================================================

SIMPLE_CUDA_ADD_KERNEL = """
extern "C" __global__
void add_kernel(const float* x, const float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}
"""

SIMPLE_CUDA_ADD_IO_CONTRACT = {
    "args": [
        {
            "name": "x",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "randn", "seed": 42}
            }
        },
        {
            "name": "y",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "randn", "seed": 43}
            }
        },
        {
            "name": "out",
            "type": "tensor",
            "role": "output",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32"
            }
        },
        {
            "name": "n",
            "type": "int",
            "value": 1024,
            "role": "input"
        }
    ],
    "launch": {
        "grid": {"x": 4, "y": 1, "z": 1},
        "block": {"x": 256, "y": 1, "z": 1}
    }
}

# ============================================================================
# MULTI-KERNEL FIXTURES
# ============================================================================

MULTI_KERNEL_EXAMPLE = """import torch
import triton
import triton.language as tl

@triton.jit
def triton_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def run_computation(x, y):
    \"\"\"Entry point for multi-kernel computation\"\"\"
    # Use Triton for addition
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    triton_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)

    # Use PyTorch for multiplication
    result = out * 2.0

    return result
"""

MULTI_KERNEL_IO_CONTRACT = {
    "args": [
        {
            "name": "x",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "randn", "seed": 42}
            }
        },
        {
            "name": "y",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "ones"}
            }
        }
    ]
}

MULTI_KERNEL_METADATA = {
    "entry_point": "run_computation",
    "description": "Multi-kernel example with Triton and PyTorch"
}

# ============================================================================
# INVALID FIXTURES (for error testing)
# ============================================================================

INVALID_TRITON_NO_IO_CONTRACT = {
    "kernel_source": """import triton
import triton.language as tl

@triton.jit
def bad_kernel(x_ptr):
    pass
""",
    "kernel_type": "triton"
    # Missing io_contract - should fail validation
}

INVALID_IO_CONTRACT_MISSING_ARGS = {
    "launch": {"grid": {"x": 4}, "num_warps": 4}
    # Missing 'args' field - should fail validation
}

INVALID_IO_CONTRACT_BAD_TENSOR_SPEC = {
    "args": [
        {
            "name": "x",
            "type": "tensor",
            "role": "input"
            # Missing tensor_spec - should fail validation
        }
    ],
    "launch": {"grid": {"x": 4}, "num_warps": 4}
}

INVALID_IO_CONTRACT_BAD_INIT_KIND = {
    "args": [
        {
            "name": "x",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "invalid_kind"}  # Invalid init kind - should fail
            }
        }
    ],
    "launch": {"grid": {"x": 4}, "num_warps": 4}
}

INVALID_IO_CONTRACT_MISSING_LAUNCH = {
    "args": [
        {
            "name": "x",
            "type": "tensor",
            "role": "input",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "zeros"}
            }
        }
    ]
    # Missing 'launch' for Triton/CUDA - should fail validation
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_kernel_fixture(kernel_type: str, variant: str = "simple"):
    """
    Get a kernel fixture by type and variant

    Args:
        kernel_type: One of "torch", "triton", "cuda", "multi_kernel"
        variant: Variant of the kernel (e.g., "simple", "matmul", "scale")

    Returns:
        Tuple of (kernel_source, io_contract, metadata)
    """
    fixtures = {
        "torch": {
            "simple": (SIMPLE_TORCH_KERNEL, None, None),
            "matmul": (TORCH_MATMUL_KERNEL, None, None)
        },
        "triton": {
            "simple": (SIMPLE_TRITON_ADD_KERNEL, SIMPLE_TRITON_ADD_IO_CONTRACT, None),
            "scale": (TRITON_VECTOR_SCALE_KERNEL, TRITON_VECTOR_SCALE_IO_CONTRACT, None)
        },
        "cuda": {
            "simple": (SIMPLE_CUDA_ADD_KERNEL, SIMPLE_CUDA_ADD_IO_CONTRACT, None)
        },
        "multi_kernel": {
            "simple": (MULTI_KERNEL_EXAMPLE, MULTI_KERNEL_IO_CONTRACT, MULTI_KERNEL_METADATA)
        }
    }

    if kernel_type not in fixtures:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    if variant not in fixtures[kernel_type]:
        raise ValueError(f"Unknown variant '{variant}' for kernel type '{kernel_type}'")

    return fixtures[kernel_type][variant]


def get_invalid_fixture(error_type: str):
    """
    Get an invalid fixture for error testing

    Args:
        error_type: Type of error to test (e.g., "no_io_contract", "missing_args")

    Returns:
        Fixture dictionary that should cause validation to fail
    """
    fixtures = {
        "no_io_contract": INVALID_TRITON_NO_IO_CONTRACT,
        "missing_args": {"io_contract": INVALID_IO_CONTRACT_MISSING_ARGS},
        "bad_tensor_spec": {"io_contract": INVALID_IO_CONTRACT_BAD_TENSOR_SPEC},
        "bad_init_kind": {"io_contract": INVALID_IO_CONTRACT_BAD_INIT_KIND},
        "missing_launch": {"io_contract": INVALID_IO_CONTRACT_MISSING_LAUNCH}
    }

    if error_type not in fixtures:
        raise ValueError(f"Unknown error type: {error_type}")

    return fixtures[error_type]
