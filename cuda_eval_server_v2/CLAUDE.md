# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kernel Evaluation Server V2** - FastAPI-based server for compiling, validating, and profiling GPU kernels with pluggable compilation backends.

**Core Features:**
- Multi-kernel support: TORCH, TORCH_CUDA, TRITON, CUDA, MULTI_KERNEL
- Subprocess isolation for safe execution
- Backend pattern architecture for extensibility
- IOContract system for explicit input/output specifications
- Optional NCU profiling for detailed GPU metrics
- Graceful failure handling (compilation/validation failures return success with error flags)
- **MCP (Model Context Protocol) support** for AI agent integration via Streamable HTTP
- Unified server architecture with both FastAPI REST API and MCP on a single port

### MCP Integration

The server supports **Model Context Protocol (MCP)** via **Streamable HTTP**, enabling AI agents to interact with the kernel evaluation system through the same unified server as the REST API.

**Running the server (includes both REST and MCP):**
```bash
# Start unified server (FastAPI REST + MCP HTTP on single port)
python main.py --host 0.0.0.0 --port 8000

# With Docker
./docker-build.sh && ./docker-run.sh --gpu all
```

**MCP Endpoint:**
- **HTTP URL**: `http://localhost:8000/mcp`
- **Transport**: Streamable HTTP (recommended MCP transport)
- **Protocol**: JSON-RPC over HTTP with bidirectional streaming

**MCP Features:**
- 5 tools for kernel compilation, validation, and profiling
  - `evaluate_kernel` - Compile, validate, and profile a single kernel
  - `compare_kernels` - Compare two kernels for correctness and performance
  - `validate_kernel` - Quick compilation and validation check
  - `get_server_stats` - Server health and GPU information
  - `get_job_status` - Check status of evaluation job
- Auto-generated schemas from Python type hints (FastMCP)
- Full integration with IOContract system
- Error handling with detailed stack traces

**MCP-specific files:**
- `mcp_server.py` - MCP server implementation (FastMCP-based)
- `tests/mcp/` - MCP-specific test suite

**Connecting MCP clients:**
```python
from mcp.client import Client
from mcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(url="http://localhost:8000/mcp")
client = Client(transport)

# Use MCP tools
result = await client.call_tool("evaluate_kernel", {
    "kernel_source": "...",
    "kernel_type": "torch"
})
```

## Development Environment

### Local vs Remote Workflow

- **Local Machine**: MacBook Pro (no GPU packages like cupy, triton installed)
  - Used for code editing only
  - Changes are rsynced to EC2 instance
- **Remote Machine**: p5e.48xlarge EC2 instance with GPU
  - Used for running server and all tests
  - Access via: `ssh p5e-cmh`
  - Working directory: `cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2`

### Python Environment on EC2

If you encounter module errors like `Error: /opt/pytorch/bin/python3: No module named pytest`:
```bash
/home/ec2-user/miniconda3/bin/conda activate base
```

## Common Commands

### Server Management

**Start server with Docker (recommended):**
```bash
# On EC2 instance
./docker-build.sh && ./docker-run.sh --gpu all

# Or restart if already running
ssh p5e-cmh "docker kill cuda-eval-server"
./docker-run.sh --gpu all
```

**Start server locally (without Docker):**
```bash
python main.py --host 0.0.0.0 --port 8000 --log-level info
```

**Health check:**
```bash
curl http://localhost:8000/health
```

### Testing

**Run test suites:**
```bash
# Unit tests - Fast, isolated component tests
pytest tests/unit/

# Integration tests - Test component interactions
pytest tests/integration/

# End-to-end tests - Full workflow validation
pytest tests/e2e/

# MCP-specific tests
pytest tests/mcp/

# Run all tests
pytest tests/

# Run specific test file
pytest tests/integration/test_endpoints.py

# Run with verbose output
pytest tests/unit/ -v
```

**Legacy test scripts** (deprecated, located in `/old_tests/`):
```bash
# These are maintained for backward compatibility but not recommended
python3 old_tests/test_server_torch_cuda.py http://localhost:8000 test-index 396
python3 old_tests/test_server_comprehensive.py --mode quick
```

## Architecture

### Request Flow

```
Client Request → FastAPI (app.py) → JobManager → Subprocess Worker
                                      ↓
                                GPU Resource Manager
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                  ↓
            CompilationService  ValidationService  ProfilingService
                    ↓
        ┌───────────┼───────────┬───────────┐
        ↓           ↓           ↓           ↓
   TorchBackend  TorchCudaBackend  TritonBackend  CudaBackend
```

### Key Components

**Frontend** (`app.py`):
- FastAPI server with async endpoints
- Routes:
  - `POST /compare` - Compare reference vs custom kernel
  - `POST /evaluate` - Evaluate single kernel
  - `GET /health` - Health check endpoint
  - `GET /stats` - Server statistics
  - `GET /job/{job_id}` - Get job status by ID
  - `POST /admin/cleanup-jobs` - Manual cleanup endpoint
  - `POST /` - Legacy endpoint (redirects to /compare)
- Handles both legacy and new request formats
- Graceful error handling with proper HTTP status codes

**Job Manager** (`orchestration/job_manager.py`):
- Orchestrates evaluation workflow
- Manages GPU resource allocation
- Spawns subprocess workers for isolation
- Handles NCU wrapping for device metrics collection
- Streams subprocess logs to main process

**Subprocess Worker** (`subprocess_worker.py`):
- Runs in complete isolation from main process
- Executes compilation → validation → profiling pipeline
- Kernel crashes don't affect server
- Communicates via temp files (`/tmp/job_{job_id}_*.json`)

**Compilation Service** (`compilation/compiler_service.py`):
- Backend pattern for pluggable compilation
- Dispatches to appropriate backend based on `kernel_type`
- Backends: `TorchBackend`, `TorchCudaBackend`, `TritonBackend`, `CudaBackend`

**Validation Service** (`validation/`):
- `CorrectnessValidator`: Compare ref vs custom kernel outputs
- `ExecutableValidator`: Verify single kernel can execute
- Tolerance-based comparison for numerical precision

**Profiling Service** (`profiling/kernel_profiler.py`):
- CUDA graphs support for accurate timing
- Statistical analysis (mean, std, median, percentiles)
- Multiple trials with warmup runs
- NVTX markers for NCU integration

### IOContract System

For TRITON and CUDA kernels, explicit input/output specifications are required:

```python
{
  "kernel": {
    "source_code": "...",
    "kernel_type": "triton",
    "io": {
      "args": [
        {
          "name": "x",
          "type": "tensor",
          "tensor_spec": {
            "shape": [1024],
            "dtype": "float32",
            "init": {"kind": "randn", "seed": 42}
          },
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
    }
  }
}
```

**IOContract Components:**
- `ArgSpec`: Defines each argument (tensors, scalars, meta parameters)
- `TensorSpec`: Shape, dtype, initialization method
- `LaunchConfig`: Grid/block dimensions for kernel launch
- Auto-generation: TORCH/TORCH_CUDA backends auto-generate from `get_inputs()`

**IOContract Helper Utilities** (`io_contract/spec_builders.py`):

The system provides helper functions to simplify IOContract creation:

```python
from io_contract.spec_builders import (
    create_randn_spec,      # Random normal distribution
    create_uniform_spec,    # Uniform distribution
    create_zeros_spec,      # All zeros
    create_ones_spec,       # All ones
    create_full_spec,       # Fill with constant value
    create_arange_spec,     # Sequential values
    IOContractBuilder       # Fluent builder for full IOContracts
)

# Example: Create tensor spec with random normal initialization
tensor_spec = create_randn_spec(
    shape=[1024, 1024],
    dtype="float32",
    seed=42
)
```

**IOContractBuilder** - Fluent API for building complete IOContracts:

```python
from io_contract.spec_builders import IOContractBuilder

contract = (
    IOContractBuilder()
    .add_input_tensor("x", shape=[1024], dtype="float32", init_kind="randn")
    .add_output_tensor("y", shape=[1024], dtype="float32")
    .add_meta_param("BLOCK_SIZE", value=256)
    .set_launch_config(grid=(4, 1, 1), num_warps=4)
    .build()
)
```

**Supported Initialization Methods:**
- `randn` - Random normal distribution (mean=0, std=1)
- `uniform` - Uniform distribution [0, 1)
- `zeros` - All zeros
- `ones` - All ones
- `full` - Fill with constant value
- `arange` - Sequential values (0, 1, 2, ...)

### Device Metrics (NCU Profiling)

Enable with environment variable:
```bash
ENABLE_DEVICE_METRICS=true python main.py
```

**How it works:**
1. JobManager wraps subprocess command with NCU
2. NVTX markers inserted with `job_id` prefix
3. NCU report saved to `/tmp/ncu_{job_id}.ncu-rep`
4. DeviceMetricsParser extracts metrics post-execution
5. Metrics attached to response in categorized structure

**Metrics Categories:**
- `speed_of_light`: Overall performance vs theoretical peak
- `memory_metrics`: DRAM bandwidth, L1/L2 hit rates
- `compute_metrics`: Pipeline utilization, occupancy limits
- `occupancy_metrics`: Warps, blocks, register usage
- `pipeline_metrics`: FMA, ALU, Tensor core activity

## Critical Design Patterns

### Graceful Failure Handling

**The server never returns HTTP 500 for kernel compilation/validation failures**. Instead:
- Compilation failure: `compiled=false`, `compilation_error="..."`, HTTP 200
- Validation failure: `correctness=false`, `validation_error="..."`, HTTP 200
- Job timeout/crash: HTTP 500 with error message

This design allows clients to distinguish between server errors and kernel errors.

### Subprocess Isolation

Every kernel evaluation runs in a subprocess:
1. **Safety**: Kernel crashes don't kill the server
2. **Isolation**: GPU state is reset between jobs
3. **Resource cleanup**: Memory leaks are contained
4. **Streaming logs**: Subprocess output streamed to main process logger

### Backend Pattern

Each kernel type has a dedicated compilation backend:
- **TorchBackend**: Pure PyTorch models, auto-generates IOContract from model structure
- **TorchCudaBackend**: PyTorch with embedded CUDA, extracts CUDA source and compiles with CuPy
- **TritonBackend**: Triton kernels, requires explicit IOContract, compiles with Triton JIT
- **CudaBackend**: Raw CUDA C++, requires IOContract, compiles with CuPy RawModule
- **MultiKernelBackend**: Python scripts with mixed kernel types, requires explicit IOContract and entry point metadata

Adding new kernel types: Inherit from `BaseCompilationBackend` and implement `compile()` method.

## Testing Philosophy

From CLAUDE.local.md: **"Always Works™" implementation**

### Core Principles
- "Should work" ≠ "does work"
- Untested code is just a guess, not a solution
- Always run the actual code to verify it works

### 30-Second Reality Check
Before claiming something is fixed:
1. Did I run/build the code?
2. Did I trigger the exact feature I changed?
3. Did I see the expected result with my own observation?
4. Did I check for error messages?
5. Would I bet $100 this works?

### Test Requirements
- **API Changes**: Make the actual API call (not just "the logic looks right")
- **Logic Changes**: Run the specific scenario
- **Integration Tests**: Start server if not running, invoke real test

### Phrases to Avoid
- "This should work now"
- "I've fixed the issue" (without testing)
- "Try it now" (without trying it myself)

## Kernel Types

| Type | Use Case | IOContract | Auto-generates Inputs |
|------|----------|------------|----------------------|
| `torch` | Pure PyTorch reference models | Optional | Yes (from model) |
| `torch_cuda` | PyTorch + embedded CUDA | Optional | Yes (from `get_inputs()`) |
| `triton` | Triton kernels | **Required** | No |
| `cuda` | Raw CUDA C++ | **Required** | No |
| `multi_kernel` | Python scripts with mixed kernel types | **Required** | No |

### Multi-Kernel Support

The `multi_kernel` type allows you to submit Python scripts that combine multiple kernel types (CUDA, Triton, Torch) in a single execution sequence.

**Requirements:**
- Must specify `entry_point` in metadata (the function name to call)
- Must provide IOContract for entry point function inputs
- Entry point function must exist in source code
- Uses package versions installed on server (Triton, CuPy, PyTorch, etc.)

**Error Handling:**
- Full stack traces returned on failures for client debugging
- Runtime errors include complete traceback

**Example Request:**
```json
{
  "kernel": {
    "source_code": "import torch\nimport triton\n...\ndef run(x, y):\n    ...",
    "kernel_type": "multi_kernel",
    "metadata": {
      "entry_point": "run",
      "description": "Fused add + multiply using Triton and PyTorch"
    },
    "io": {
      "args": [
        {
          "name": "x",
          "type": "tensor",
          "tensor_spec": {
            "shape": [1024],
            "dtype": "float32",
            "init": {"kind": "randn", "seed": 42}
          },
          "role": "input"
        }
      ]
    }
  }
}
```

## Common Development Tasks

### Adding a New Compilation Backend

1. Create backend file in `compilation/{backend_name}/`
2. Inherit from `BaseCompilationBackend`
3. Implement `compile(request: CompilationRequest, gpu_id: int) -> CompilationResult`
4. Register in `CompilationService.__init__()`:
   ```python
   self.backends[KernelType.NEW_TYPE] = NewBackend()
   ```
5. Add enum to `KernelType` in `shared/models.py`

### Debugging Compilation Failures

1. Check compilation error in response: `compilation_error` field
2. Enable debug logging: `LOG_LEVEL=DEBUG python main.py`
3. Check subprocess logs: Search for `worker.{job_id}` in logs
4. Inspect temp files if subprocess crashed: `/tmp/job_{job_id}_*.json`

### Testing New Features

1. **Always test on EC2**, not local Mac (GPU packages required)
2. **Start server in Docker** for integration tests
3. **Run actual API calls**, not just unit tests
4. **Check both success and failure cases**
5. **Verify logs** for expected behavior

## File Structure

```
cuda_eval_server_v2/
├── app.py                      # FastAPI application
├── main.py                     # Server entry point
├── mcp_server.py               # MCP server implementation
├── subprocess_worker.py        # Isolated worker process
│
├── orchestration/
│   └── job_manager.py          # Job orchestration, GPU management
│
├── compilation/
│   ├── compiler_service.py     # Backend dispatcher
│   ├── torch/                  # Pure PyTorch backend
│   ├── torch_cuda/             # PyTorch + CUDA backend
│   ├── triton/                 # Triton backend (includes TritonExecutableKernel)
│   ├── cuda/                   # Raw CUDA backend
│   └── multi_kernel/           # Multi-kernel backend
│
├── validation/
│   ├── correctness_validator.py  # Compare kernels, ExecutableValidator
│   └── base_validator.py         # Abstract validator base class
│
├── profiling/
│   └── kernel_profiler.py      # CUDA graphs, statistical analysis
│
├── shared/
│   ├── models.py               # Data models (Pydantic + dataclasses)
│   ├── executable_kernels.py  # Executable kernel wrappers (Torch, TorchCuda, Cuda, MultiKernel)
│   ├── kernel_metadata.py     # Kernel metadata classes
│   ├── metrics_collector.py   # Metrics collection system
│   ├── gpu_resource_manager.py # GPU allocation
│   ├── device_metrics_parser.py # NCU report parsing
│   └── utils.py                # Utilities
│
├── io_contract/
│   ├── manager.py              # IOContract management
│   ├── tensor_utils.py         # Tensor generation utilities
│   └── spec_builders.py        # Helper builders for IOContract specs
│
├── client/                     # Python client library
│   ├── kernel_client.py        # Client for server API
│   ├── examples/               # Example usage scripts
│   └── setup.py                # Client package setup
│
├── tests/
│   ├── unit/                   # Fast, isolated component tests
│   ├── integration/            # Component interaction tests
│   ├── e2e/                    # End-to-end workflow tests
│   ├── mcp/                    # MCP-specific tests
│   └── fixtures/               # Test fixtures and factories
│
├── old_tests/                  # Legacy test scripts (DEPRECATED)
│
├── scripts/
│   ├── docker-build-mcp.sh     # Build MCP Docker image
│   ├── docker-run-mcp.sh       # Run MCP Docker container
│   ├── start-mcp.sh            # Start MCP server
│   └── start-hybrid.sh         # Start hybrid mode server
│
├── docker-build.sh             # Build FastAPI Docker image
├── docker-run.sh               # Run FastAPI Docker container
├── Dockerfile                  # Base Docker image
├── Dockerfile.optimized        # Optimized Docker image
├── Dockerfile.mcp              # MCP Docker image
├── docker-compose.yml          # Docker compose for FastAPI
└── docker-compose.mcp.yml      # Docker compose for MCP
```

## Server Configuration

### Command-Line Arguments

The server is configured via command-line arguments in `main.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | 0.0.0.0 | Server host address |
| `--port` | 8000 | Server port |
| `--log-level` | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `--mode` | fastapi | Server mode: "fastapi", "mcp", or "both" (hybrid) |
| `--reload` | false | Enable auto-reload for development |
| `--cupy-cache-dir` | /tmp/cupy_kernel_cache | CuPy kernel compilation cache directory |

**Example:**
```bash
python main.py --host 0.0.0.0 --port 8000 --log-level DEBUG --mode both
```

### Environment Variables

| Variable | Usage | Description |
|----------|-------|-------------|
| `GPU_TYPE` | Optional | GPU type detection in `app.py` |
| `CUDA_LAUNCH_BLOCKING` | Set by profiler | Forces synchronous CUDA operations for profiling |
| `TORCH_USE_CUDA_DSA` | Set by profiler | Enables CUDA device-side assertions |
| `CUPY_KERNEL_CACHE_DIR` | Optional | Can also be set via environment (overridden by CLI arg) |

## Related Documentation

- `README.md` - Quick start and API overview
- `USER_MANUAL.md` - Detailed system design
- `API_GUIDE.md` - Complete API reference with examples
- `TEST_GUIDE.md` - Comprehensive testing guide
- `DEVICE_METRICS_GUIDE.md` - NCU profiling usage
- `DEPLOYMENT_CLUSTER.md` - Kubernetes deployment
