# CUDA Evaluation Server V2 - Backend-Based Kernel Compilation & Profiling

🚀 **Flexible kernel evaluation server with pluggable compilation backends for multiple kernel types**

## Overview

This is a refactored version of the CUDA kernel evaluation server featuring:

- **Backend Pattern Architecture**: Pluggable compilation backends for different kernel types
- **Multi-Kernel Support**: TORCH, TORCH_CUDA, TRITON, and CUDA kernels (CUDA in development)
- **FastAPI + Async**: Non-blocking request handling with subprocess isolation
- **IOContract Integration**: Comprehensive input/output specifications embedded in KernelCode
- **Tensor Transfer Support**: Transfer tensors over FastAPI with compression or generate them server-side
- **Triton Kernel Support**: Full support for Triton kernel compilation and execution
- **Subprocess Safety**: Isolated execution prevents kernel crashes from affecting the server
- **Device Metrics Collection**: Optional NCU profiling for detailed GPU performance analysis
- **Extensible Design**: Easy to add new kernel types and compilation backends

## Architecture

```
┌─────────┐    ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Request │───▶│   FastAPI    │───▶│   JobManager     │───▶│   Subprocess    │
│         │    │   (app.py)   │    │ (Orchestration)  │    │     Worker      │
└─────────┘    └──────────────┘    └──────────────────┘    └─────────────────┘
                                            │                        │
                                            ▼                        ▼
                                    ┌───────────────┐      ┌─────────────────┐
                                    │ GPU Resource  │      │  Compilation    │
                                    │   Manager     │      │    Service      │
                                    └───────────────┘      └─────────────────┘
                                                                     │
                                                            ┌────────┴────────┐
                                                            ▼                 ▼
                                                    ┌──────────────┐  ┌──────────────┐
                                                    │ TorchCuda    │  │    Torch     │
                                                    │   Backend    │  │   Backend    │
                                                    └──────────────┘  └──────────────┘
```

### Components

1. **Frontend** (`app.py`): FastAPI server with async request handlers
2. **Job Manager** (`orchestration/job_manager.py`): Orchestrates evaluation workflow using subprocess workers
3. **Compilation Service** (`compilation/compiler_service.py`): Backend-based kernel compilation
4. **Compilation Backends**:
   - `TorchCudaCompilationBackend`: For PyTorch models with embedded CUDA
   - `TorchCompilationBackend`: For pure PyTorch reference models
   - `TritonCompilationBackend`: For Triton kernel implementations
5. **Validation Service** (`validation/`): Kernel correctness validation
6. **Profiling Service** (`profiling/kernel_profiler.py`): Performance measurement with CUDA graphs support
7. **Subprocess Worker** (`subprocess_worker.py`): Isolated execution environment

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install fastapi>=0.104.0 uvicorn[standard]>=0.24.0
pip install torch>=2.6.0 numpy>=1.21.0

# Optional: For C++ wrapper transformation
pip install libclang>=16.0.0

# Optional: NVIDIA Nsight Compute for device metrics
# Download from: https://developer.nvidia.com/nsight-compute
```

### Running the Server

```bash
# Navigate to server directory
cd AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2/

# Basic launch
python main.py

# With options
python main.py --host 0.0.0.0 --port 8000 --log-level info

# With NCU device metrics
ENABLE_DEVICE_METRICS=true python main.py
```

## API Usage

### Primary Endpoint

The server supports both legacy and new API formats:

#### New Format (Recommended)
```bash
curl -X POST "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_kernel": {
      "source_code": "...",
      "kernel_type": "torch"
    },
    "custom_kernel": {
      "source_code": "...",
      "kernel_type": "torch_cuda"
    },
    "num_trials": 100,
    "timeout": 120
  }'
```

#### Legacy Format (Backward Compatible)
```bash
curl -X POST "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_code": "...",
    "custom_code": "...",
    "num_trials": 100
  }'
```

### Kernel Types

| Type | Description | Example Use | Status |
|------|-------------|-------------|--------|
| `TORCH` | Pure PyTorch models | Reference implementations | ✅ Fully Supported |
| `TORCH_CUDA` | PyTorch with embedded CUDA | Generated kernels with load_inline | ✅ Fully Supported |
| `TRITON` | Triton kernels | Triton kernel implementations | ✅ Fully Supported |
| `CUDA` | Raw CUDA kernels | Direct CUDA code | 🚧 In Development |

### Health Check

```bash
curl http://localhost:8000/health
```

### Additional Endpoints

```bash
# Get job status
curl http://localhost:8000/job/{job_id}

# Get server statistics
curl http://localhost:8000/stats

# Admin: Cleanup old jobs
curl -X POST http://localhost:8000/admin/cleanup-jobs
```

## Device Metrics Collection

The server supports NCU profiling for detailed GPU performance analysis:

```bash
# Enable device metrics
export ENABLE_DEVICE_METRICS=true

# Configure NCU sections
export NCU_SECTIONS="SpeedOfLight,Occupancy,ComputeWorkloadAnalysis"

# Run server with metrics
python main.py
```

See [DEVICE_METRICS_GUIDE.md](DEVICE_METRICS_GUIDE.md) for detailed documentation.

## Compilation Backends

The server uses a backend pattern for compilation, allowing different approaches for different kernel types:

### TorchCudaCompilationBackend
- Handles PyTorch models with embedded CUDA (using `load_inline`)
- Extracts CUDA kernels and C++ wrappers
- Transforms C++ code for compatibility
- Compiles using PyTorch's torch.utils.cpp_extension

### TorchCompilationBackend
- Handles pure PyTorch reference models
- Direct execution without compilation
- Used for baseline performance comparison

### TritonCompilationBackend
- Handles Triton kernel implementations
- Supports runtime capture and IOContract-based execution
- Manages Triton-specific meta parameters and grid configurations

### Adding New Backends

To add support for new kernel types:

1. Create a new backend class inheriting from `BaseCompilationBackend`
2. Implement the `compile()` method
3. Register in `CompilationService.backends`

## Subprocess Isolation

The server uses subprocess workers for safety:
- Kernel compilation and validation run in isolated processes
- Segfaults and crashes don't affect the main server
- GPU resources properly managed across processes
- Log streaming from subprocess to main process

## Project Structure

```
cuda_eval_server_v2/
├── app.py                          # FastAPI frontend
├── main.py                         # Entry point
├── subprocess_worker.py            # Isolated execution worker
├── orchestration/
│   └── job_manager.py             # Job orchestration
├── compilation/
│   ├── compiler_service.py        # Backend-based compilation
│   ├── base_compiler.py           # Base backend interface
│   ├── torch/                     # Pure PyTorch backend
│   │   └── torch_backend.py
│   ├── torch_cuda/                # PyTorch+CUDA backend
│   │   ├── torch_cuda_backend.py
│   │   ├── compiler.py
│   │   ├── kernel_extractor.py
│   │   └── cpp_wrapper_transformer.py
│   └── triton/                    # Triton backend
│       └── triton_backend.py
├── validation/
│   ├── base_validator.py          # Validation interface
│   └── torch_cuda_validator.py    # TORCH_CUDA validation
├── profiling/
│   └── kernel_profiler.py         # Performance profiling
└── shared/
    ├── models.py                   # Data models with IOContract
    ├── utils.py                    # Tensor encoding/decoding utilities
    ├── executable_kernels.py       # Kernel execution abstraction
    ├── args_generator.py           # Input generation
    ├── device_metrics_parser.py    # NCU metrics parsing
    └── metrics_collector.py        # Performance metrics
```

## Performance Comparison

| Component | Old Architecture | New Architecture |
|-----------|-----------------|------------------|
| **Design** | CuPy-centric | Strategy pattern |
| **Kernel Types** | CUDA only | Multiple types |
| **Compilation** | Single approach | Pluggable strategies |
| **Safety** | In-process | Subprocess isolation |
| **Extensibility** | Limited | Highly extensible |
| **API** | Simple strings | Typed kernel objects |

## Migration from V1

The server maintains backward compatibility with the V1 API while offering enhanced functionality through the new API. See the API Usage section for examples of both formats.

### Key Changes

1. **API Format**: Now uses `KernelCode` objects with explicit `kernel_type`
2. **Compilation**: Strategy-based instead of CuPy-only
3. **File Names**: Several modules renamed (see migration table below)

| Old Name | New Name/Location |
|----------|-------------------|
| `simple_cupy_compiler.py` | Use compilation strategies |
| `separated_profiler.py` | `profiling/kernel_profiler.py` |
| `input_generator.py` | `shared/args_generator.py` |
| `subprocess_correctness_validator.py` | `validation/torch_cuda_validator.py` |

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'compilation'
   ```
   Solution: Ensure you're running from the correct directory

2. **Kernel Type Not Supported**
   ```
   ValueError: No compilation strategy available for kernel type
   ```
   Solution: Check that kernel_type is one of: torch, torch_cuda, cuda, triton

3. **Subprocess Failures**
   ```
   Subprocess completed with exit code 1
   ```
   Solution: Check subprocess worker logs for detailed error messages

4. **GPU Resource Issues**
   ```
   TimeoutError: No GPU available within 300 seconds
   ```
   Solution: Check GPU availability with `nvidia-smi`

## Future Enhancements

- [ ] Add CUDA compilation strategy for raw CUDA kernels
- [ ] Add Triton compilation strategy
- [ ] Implement compilation caching across requests
- [ ] Add distributed evaluation support
- [ ] Enhanced metrics dashboard

## Contributing

1. **Add Strategies**: Extend `BaseCompilationStrategy` for new kernel types
2. **Add Validators**: Extend `BaseValidator` for validation logic
3. **Maintain API**: Keep backward compatibility when possible
4. **Add Tests**: Test each component independently

---

**Production Ready**: The server provides a flexible, extensible architecture for evaluating various kernel types with robust error handling and performance profiling.
