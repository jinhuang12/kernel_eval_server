# MCP Server for CUDA Evaluation

## Overview

The CUDA Evaluation Server V2 now supports the **Model Context Protocol (MCP)**, allowing Claude to directly interact with GPU kernel evaluation capabilities. This enables Claude to compile, validate, and profile GPU kernels without requiring HTTP requests.

## Architecture

The server supports three operational modes:

1. **FastAPI mode** (`--mode fastapi`): Original REST API server (HTTP)
2. **MCP mode** (`--mode mcp`): MCP server via stdio protocol
3. **Hybrid mode** (`--mode both`): Both FastAPI and MCP running simultaneously

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Claude Desktop  →  MCP Server  →  JobManager  │
│                         ↓                       │
│                   GPU Services                  │
│                 (Compile/Profile)               │
│                                                 │
└─────────────────────────────────────────────────┘
```

## MCP Tools

The MCP server exposes the following tools:

### 1. `evaluate_kernel`

Compile, validate, and profile a GPU kernel with subprocess isolation for safety.

**Parameters:**
- `kernel_source` (required): Source code of the kernel
- `kernel_type` (required): `torch`, `torch_cuda`, `triton`, `cuda`, or `multi_kernel`
- `io_contract` (optional): Input/output specification (REQUIRED for Triton/CUDA/MULTI_KERNEL)
- `metadata` (optional): Kernel metadata for advanced targeting:
  - TORCH: `{"function_name": "...", "class_name": "...", "method_name": "..."}`
  - TRITON: `{"kernel_name": "..."}` for selecting specific kernel
  - MULTI_KERNEL: `{"entry_point": "..."}` (REQUIRED - specifies function to call)
  - CUDA: `{"kernel_name": "...", "compile_options": [...]}`
- `reference_kernel_source` (optional): Reference kernel for comparison mode
- `reference_kernel_type` (optional): Type of reference kernel
- `reference_kernel_metadata` (optional): Metadata for reference kernel
- `num_trials` (default: 100): Number of profiling trials
- `num_warmup` (default: 10): Number of warmup runs before profiling
- `atol` (default: 1e-8): Absolute tolerance for validation
- `rtol` (default: 1e-5): Relative tolerance for validation
- `timeout` (default: 120): Maximum execution time in seconds
- `enable_device_metrics` (default: false): Enable NCU profiling (adds significant overhead)

**Returns:**
- Compilation status
- Validation results
- Runtime statistics (mean, std, min, max, median, percentiles)
- Device metrics (if enabled)

### 2. `validate_kernel`

Compile and validate a kernel without profiling. More useful than compile-only as it catches runtime errors that only appear during execution.

**Parameters:**
- `kernel_source` (required): Source code of the kernel
- `kernel_type` (required): `torch`, `torch_cuda`, `triton`, `cuda`, or `multi_kernel`
- `io_contract` (optional): Required for Triton/CUDA/MULTI_KERNEL
- `metadata` (optional): Kernel metadata for advanced targeting (e.g., entry_point for MULTI_KERNEL)
- `timeout` (default: 60): Maximum execution time in seconds

**Returns:**
- `compiled`: Whether kernel compiled successfully
- `validated`: Whether kernel executed without errors
- `compilation_error`: Error message if compilation failed
- `validation_error`: Error message if validation failed
- `compilation_time`: Time taken to compile
- `status`: Overall status of the operation

**Why validation over compile-only:** JIT-compiled kernels (Triton, CUDA) may have runtime errors that only appear during execution, such as out-of-bounds memory access, incorrect grid dimensions, or missing meta parameters.

### 3. `compare_kernels`

Compare the performance and correctness of two GPU kernels (reference vs custom). This is a dedicated tool for kernel comparison, separate from `evaluate_kernel`.

**Parameters:**
- `ref_kernel_source` (required): Source code of the reference kernel
- `ref_kernel_type` (required): Type of reference kernel (default: `torch`)
- `ref_kernel_io` (optional): IOContract for reference kernel
- `ref_kernel_metadata` (optional): Metadata for reference kernel
- `custom_kernel_source` (required): Source code of the custom kernel
- `custom_kernel_type` (required): Type of custom kernel
- `custom_kernel_io` (optional): IOContract for custom kernel (required for Triton/CUDA/MULTI_KERNEL)
- `custom_kernel_metadata` (optional): Metadata for custom kernel
- `num_trials` (default: 100): Number of profiling trials
- `num_warmup` (default: 10): Number of warmup runs before profiling
- `atol` (default: 1e-5): Absolute tolerance for correctness validation
- `rtol` (default: 1e-5): Relative tolerance for correctness validation
- `timeout` (default: 120): Maximum execution time in seconds

**Returns:**
- Compilation status for both kernels
- Correctness validation (outputs compared with tolerance)
- Performance comparison with runtime statistics for both kernels
- Speedup factor (custom vs reference)
- Device metrics (if enabled)

**When to use:** Use this tool when you need to compare two kernel implementations for correctness and performance. Both kernels receive identical inputs for fair comparison.

### 4. `get_server_stats`

Get server health and GPU information.

**Returns:**
- Server status
- GPU availability and device info
- Memory usage
- Job statistics

### 5. `get_job_status`

Poll status of a running job.

**Parameters:**
- `job_id` (required): Job identifier

**Returns:**
- Current status (submitted, compiling, validating, profiling, completed, failed)
- Creation time
- Error message (if failed)

## Setup

### Prerequisites

- CUDA 12.x with compatible GPU
- Python 3.12
- Docker (optional, for containerized deployment)

### Local Setup (MacBook → EC2)

Since GPU packages aren't available on MacBook, the MCP server runs on EC2 via SSH.

#### 1. Install MCP Package on EC2

```bash
# SSH to EC2
ssh p5e-cmh

# Activate conda environment
/home/ec2-user/miniconda3/bin/conda activate base

# Navigate to server directory
cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2

# Install MCP dependency
pip install mcp>=0.9.0
```

#### 2. Configure Claude Desktop

Add to Claude Desktop's MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "cuda-eval-server-ec2": {
      "command": "ssh",
      "args": [
        "p5e-cmh",
        "cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2 && /home/ubuntu/miniconda3/bin/conda run -n base python main.py --mode mcp --log-level info"
      ],
      "env": {
        "GPU_DEVICE": "0",
        "ENABLE_DEVICE_METRICS": "false",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

#### 3. Test Connection

Restart Claude Desktop and verify the MCP server appears in the tools list.

### Docker Setup (EC2)

For containerized deployment on EC2:

#### 1. Build MCP Docker Image

```bash
cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2

# Build MCP-specific image
docker build -f Dockerfile.mcp -t cuda-eval-mcp-server:latest ../../../
```

#### 2. Run MCP Server

**MCP mode only:**
```bash
docker run --gpus all --rm -it \
  -e GPU_DEVICE=0 \
  -e ENABLE_DEVICE_METRICS=false \
  -v /tmp:/tmp \
  cuda-eval-mcp-server:latest
```

**Hybrid mode (FastAPI + MCP):**
```bash
docker run --gpus all --rm -it \
  -p 8000:8000 \
  -e GPU_DEVICE=0 \
  -e ENABLE_DEVICE_METRICS=false \
  -v /tmp:/tmp \
  cuda-eval-mcp-server:latest \
  python3 main.py --mode both --host 0.0.0.0 --port 8000 --log-level info
```

#### 3. Using Docker Compose

```bash
# Start MCP server
docker-compose -f docker-compose.mcp.yml up mcp-server

# Start hybrid mode
docker-compose -f docker-compose.mcp.yml up hybrid-server

# Start FastAPI only
docker-compose -f docker-compose.mcp.yml up fastapi-server
```

## IOContract Quick Reference

The **IOContract** system defines kernel inputs, outputs, and launch configuration. It's **REQUIRED** for Triton, CUDA, and Multi-Kernel types, but **optional** for Torch/TorchCuda (auto-generated if omitted).

### Basic Structure

```json
{
  "args": [...],     // Array of argument specifications
  "launch": {...}    // Kernel launch configuration
}
```

### Argument Types

#### Input Tensor

```json
{
  "name": "x",
  "type": "tensor",
  "role": "input",
  "tensor_spec": {
    "shape": [1024],
    "dtype": "float32",
    "init": {"kind": "randn", "seed": 42}
  }
}
```

#### Output Tensor

```json
{
  "name": "result",
  "type": "tensor",
  "role": "output",
  "tensor_spec": {
    "shape": [1024],
    "dtype": "float32"
  }
}
```

#### Scalar Argument (int, float, str, bool)

```json
{
  "name": "n",
  "type": "int",
  "value": 1024,
  "role": "input"
}
```

#### Triton Compile-Time Constant (Meta Parameter)

```json
{
  "name": "BLOCK_SIZE",
  "type": "int",
  "value": 256,
  "role": "input",
  "is_meta": true
}
```

**Note:** `is_meta=true` is required for Triton's `tl.constexpr` parameters (template parameters known at compile time).

### Initialization Methods

| Method | Parameters | Description | Example |
|--------|-----------|-------------|---------|
| `randn` | `seed`, `mean`, `std` | Normal distribution (Gaussian) | `{"kind": "randn", "seed": 42}` |
| `uniform` | `seed`, `low`, `high` | Uniform distribution | `{"kind": "uniform", "low": 0.0, "high": 1.0}` |
| `zeros` | None | All zeros | `{"kind": "zeros"}` |
| `ones` | None | All ones | `{"kind": "ones"}` |
| `full` | `fill_value` | Constant value | `{"kind": "full", "fill_value": 3.14}` |
| `arange` | `start`, `step` | Sequential values | `{"kind": "arange", "start": 0, "step": 1}` |

### Launch Configuration

#### Triton Kernels

```json
{
  "grid": {"x": 4, "y": 1, "z": 1},
  "num_warps": 4,           // Warps per block (1, 2, 4, 8, 16, 32)
  "num_stages": 3           // Pipeline stages (optional, default 3)
}
```

**Typical values:**
- `num_warps`: 4 or 8 (must be power of 2)
- `grid.x`: Number of blocks = `ceil(data_size / BLOCK_SIZE)`

#### CUDA Kernels

```json
{
  "grid": {"x": 16, "y": 1, "z": 1},     // Number of blocks
  "block": {"x": 256, "y": 1, "z": 1}    // Threads per block
}
```

**Typical values:**
- `block.x`: 256-1024 threads per block
- `grid.x`: `ceil(data_size / block.x)`

### Data Types

**Supported dtypes:** `float32`, `float64`, `float16`, `bfloat16`, `int32`, `int64`, `int8`, `uint8`, `bool`

### Argument Roles

| Role | Description | Used For |
|------|-------------|----------|
| `input` | Read by kernel | Input tensors/scalars |
| `output` | Written by kernel | Output tensors |
| `inout` | Read and written | In-place operations |

### Common Patterns

#### 2D Tensor with Uniform Initialization

```json
{
  "name": "matrix",
  "type": "tensor",
  "role": "input",
  "tensor_spec": {
    "shape": [1024, 512],
    "dtype": "float32",
    "init": {"kind": "uniform", "low": -1.0, "high": 1.0, "seed": 123}
  }
}
```

#### Multiple Output Tensors

```json
{
  "args": [
    {"name": "x", "type": "tensor", "role": "input", "tensor_spec": {...}},
    {"name": "out1", "type": "tensor", "role": "output", "tensor_spec": {...}},
    {"name": "out2", "type": "tensor", "role": "output", "tensor_spec": {...}}
  ]
}
```

#### Inout Tensor (In-Place Operation)

```json
{
  "name": "data",
  "type": "tensor",
  "role": "inout",
  "tensor_spec": {
    "shape": [2048],
    "dtype": "float32",
    "init": {"kind": "randn", "seed": 42}
  }
}
```

## Usage Examples

### Example 1: Evaluate a Simple Torch Kernel

```json
{
  "kernel_source": "import torch\n\nclass Model(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = torch.nn.Linear(10, 10)\n    \n    def forward(self, x):\n        return self.fc(x)\n\ndef get_inputs():\n    return [torch.randn(32, 10)]",
  "kernel_type": "torch",
  "num_trials": 100
}
```

### Example 2: Compare Two Kernels (Using compare_kernels Tool)

```json
{
  "ref_kernel_source": "import torch\n\nclass ReferenceMatmul(torch.nn.Module):\n    def forward(self, x, y):\n        return torch.mm(x, y)\n\ndef get_inputs():\n    return [torch.randn(64, 128), torch.randn(128, 256)]",
  "ref_kernel_type": "torch",
  "custom_kernel_source": "import torch\n\nclass CustomMatmul(torch.nn.Module):\n    def forward(self, x, y):\n        return torch.matmul(x, y)\n\ndef get_inputs():\n    return [torch.randn(64, 128), torch.randn(128, 256)]",
  "custom_kernel_type": "torch",
  "num_trials": 100,
  "atol": 1e-5,
  "rtol": 1e-5
}
```

### Example 3: Evaluate a Triton Kernel

```json
{
  "kernel_source": "import triton\nimport triton.language as tl\n\n@triton.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n    mask = offs < n\n    x = tl.load(x_ptr + offs, mask=mask)\n    y = tl.load(y_ptr + offs, mask=mask)\n    tl.store(out_ptr + offs, x + y, mask=mask)",
  "kernel_type": "triton",
  "io_contract": {
    "args": [
      {
        "name": "x_ptr",
        "type": "tensor",
        "tensor_spec": {
          "shape": [1024],
          "dtype": "float32",
          "init": {"kind": "randn", "seed": 42}
        },
        "role": "input"
      },
      {
        "name": "y_ptr",
        "type": "tensor",
        "tensor_spec": {
          "shape": [1024],
          "dtype": "float32",
          "init": {"kind": "randn", "seed": 43}
        },
        "role": "input"
      },
      {
        "name": "out_ptr",
        "type": "tensor",
        "tensor_spec": {
          "shape": [1024],
          "dtype": "float32",
          "init": {"kind": "zeros"}
        },
        "role": "output"
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
        "is_meta": true
      }
    ],
    "launch": {
      "grid": {"x": 4, "y": 1, "z": 1},
      "num_warps": 4
    }
  },
  "num_trials": 100
}
```

### Example 4: Validate a Kernel (Quick Testing Without Profiling)

Use `validate_kernel` for rapid iteration during development - it compiles and validates without the overhead of profiling.

```json
{
  "kernel_source": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef my_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n    x = tl.load(x_ptr + offs)\n    tl.store(x_ptr + offs, x * 2)",
  "kernel_type": "triton",
  "io_contract": {
    "args": [
      {
        "name": "x_ptr",
        "type": "tensor",
        "role": "inout",
        "tensor_spec": {
          "shape": [1024],
          "dtype": "float32",
          "init": {"kind": "randn", "seed": 42}
        }
      },
      {
        "name": "BLOCK_SIZE",
        "type": "int",
        "value": 256,
        "role": "input",
        "is_meta": true
      }
    ],
    "launch": {
      "grid": {"x": 4},
      "num_warps": 4
    }
  }
}
```

### Example 5: Using Metadata for MULTI_KERNEL with Entry Point

The `metadata` parameter is REQUIRED for MULTI_KERNEL to specify which function to call.

```json
{
  "kernel_source": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    mask = offs < n\n    x = tl.load(x_ptr + offs, mask=mask)\n    y = tl.load(y_ptr + offs, mask=mask)\n    tl.store(out_ptr + offs, x + y, mask=mask)\n\ndef run(x, y):\n    '''Entry point that mixes Triton and PyTorch'''\n    output = torch.empty_like(x)\n    n = x.numel()\n    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)\n    add_kernel[grid](x, y, output, n, BLOCK=256)\n    return output * 2  # PyTorch operation",
  "kernel_type": "multi_kernel",
  "metadata": {
    "entry_point": "run"
  },
  "io_contract": {
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
  },
  "num_trials": 50
}
```

### Example 6: Using Metadata for TORCH Function Targeting

Target specific functions or methods in TORCH kernels using metadata.

```json
{
  "kernel_source": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return x * 2\n\ndef custom_function(x):\n    '''Standalone function to target'''\n    return torch.relu(x) + 1\n\nclass CustomModel(nn.Module):\n    def forward(self, x):\n        return x * 3\n    \n    def custom_method(self, x):\n        '''Specific method to target'''\n        return torch.sigmoid(x)",
  "kernel_type": "torch",
  "metadata": {
    "class_name": "CustomModel",
    "method_name": "custom_method"
  },
  "io_contract": {
    "args": [
      {
        "name": "x",
        "type": "tensor",
        "role": "input",
        "tensor_spec": {
          "shape": [32, 32],
          "dtype": "float32",
          "init": {"kind": "randn"}
        }
      }
    ]
  },
  "num_trials": 100
}
```

## Testing

### Unit Tests

Run unit tests (no GPU required, tests MCP server structure):

```bash
cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2
pytest tests/mcp/test_mcp_tools.py -v
```

### Integration Tests

Run integration tests on EC2 (requires GPU):

```bash
# Activate conda environment
/home/ubuntu/miniconda3/bin/conda activate base

# Run integration tests
pytest tests/mcp/test_mcp_integration.py -v -m ec2_only
```

### Manual Testing

Test MCP server manually:

```bash
# Start MCP server
python main.py --mode mcp --log-level info

# In another terminal, send test request via stdin
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | python main.py --mode mcp
```

## Troubleshooting

### Issue: MCP server not appearing in Claude Desktop

**Solution:**
1. Check Claude Desktop configuration file location
2. Verify SSH connection to EC2 works: `ssh p5e-cmh`
3. Check server logs for errors
4. Restart Claude Desktop after configuration changes

### Issue: Import errors (cupy, triton not found)

**Solution:**
```bash
# Activate conda environment first
/home/ubuntu/miniconda3/bin/conda activate base

# Verify packages installed
python -c "import cupy; import triton; import torch; print('All packages OK')"
```

### Issue: GPU not available

**Solution:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Issue: Permission denied for /tmp files

**Solution:**
```bash
# Fix permissions
sudo chmod -R 777 /tmp/cupy_kernel_cache
sudo chmod -R 777 /tmp/torch_extensions
```

## Performance Considerations

### Device Metrics Collection

NCU profiling adds significant overhead (~10-100x slower). Only enable when detailed metrics are needed:

```json
{
  "enable_device_metrics": true
}
```

### Profiling Trials

Balance accuracy vs speed:
- **Quick test**: `num_trials=10` (~1-5 seconds)
- **Standard**: `num_trials=100` (~5-30 seconds)
- **High accuracy**: `num_trials=1000` (~30-300 seconds)

### Timeout Settings

Set appropriate timeouts based on kernel complexity:
- **Simple kernels**: `timeout=30`
- **Complex kernels**: `timeout=120`
- **Large models**: `timeout=300`

## Advanced Configuration

### Custom GPU Selection

```bash
# Use specific GPU
export GPU_DEVICE=1
python main.py --mode mcp
```

### Enable Debug Logging

```bash
python main.py --mode mcp --log-level debug
```

### Custom Cache Directory

```bash
export CUPY_KERNEL_CACHE_DIR=/path/to/cache
python main.py --mode mcp
```

## Architecture Details

### Subprocess Isolation

Each kernel evaluation runs in an isolated subprocess:
- Kernel crashes don't affect the server
- GPU state is reset between jobs
- Memory leaks are contained

### Job Management

Jobs are tracked through states:
1. `submitted`: Job received
2. `compiling`: Kernel compilation in progress
3. `validating`: Correctness validation running
4. `profiling`: Performance profiling in progress
5. `completed`: Job finished successfully
6. `failed`: Job encountered an error

### Graceful Failure Handling

The server never crashes on kernel errors:
- Compilation failures: Return `compiled=false` with error message
- Validation failures: Return `correctness=false` with error message
- Profiling failures: Return partial results with error message

## Contributing

To add new MCP tools:

1. Add tool definition in `mcp_server.py` `list_tools()` method
2. Implement handler method (e.g., `_my_new_tool()`)
3. Register in `call_tool()` dispatcher
4. Add tests in `tests/mcp/test_mcp_tools.py`
5. Update this documentation

## Support

For issues or questions:
- Check existing issues: https://github.com/anthropics/claude-code/issues
- Server logs: Check stdout/stderr from MCP server
- Job-specific issues: Use `get_job_status` tool for debugging
