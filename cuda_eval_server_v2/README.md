# CUDA Evaluation Server V2

**Compile, validate, and profile GPU kernels with a simple REST API**

## What is this?

A production-ready server that evaluates GPU kernel performance. Send your kernel code, get back detailed performance metrics.

### The Problem
- Writing optimized GPU kernels is hard
- Testing performance requires complex boilerplate code  
- Comparing different implementations is tedious
- Profiling needs specialized tools and expertise

### The Solution
This server handles all the complexity. You send code, we handle:
- ✅ Compilation across different frameworks (PyTorch, CUDA, Triton)
- ✅ Performance profiling with statistical analysis
- ✅ Correctness validation between implementations
- ✅ Detailed GPU metrics collection (optional NCU profiling)
- ✅ Safe execution in isolated subprocesses

## Quick Start (5 minutes)

### System Requirements
- **GPU**: NVIDIA GPU with CUDA Capability 8.0+ (A100, H100, H200 etc.)
- **CUDA**: Version 12.0 or higher
- **Python**: 3.11+
- **OS**: Linux (Ubuntu 22.04+ recommended)

### Option 1: Docker (Recommended)

```bash
# Pull and run the server
docker pull 592892253131.dkr.ecr.us-east-1.amazonaws.com/cuda-eval-server:latest
docker run -it --rm --name cuda-eval-server --user root --cap-add=SYS_ADMIN --security-opt seccomp=unconfined -p 8000:8000 --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e PYTHONUNBUFFERED=1 -e ENABLE_DEVICE_METRICS=true -e LOG_LEVEL=info 592892253131.dkr.ecr.us-east-1.amazonaws.com/cuda-eval-server:latest

# Or build & run it from source
./docker-build.sh && ./docker-run.sh --gpu all

# Verify it's running
curl http://localhost:8000/health
# Expected: {"status": "healthy", "gpu_available": true, "gpu_count": 1}
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/KernelBench.git
cd KernelBench/scripts/cuda_eval_server_v2

# Install dependencies
pip install -r ../../../requirements.eval_server_v2.txt

# Start the server
python main.py

# Verify installation
curl http://localhost:8000/health
```

### Your First Request

Evaluate a simple PyTorch kernel:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "kernel": {
      "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return torch.relu(x)",
      "kernel_type": "torch"
    },
    "num_trials": 100
  }'
```

Expected response:
```json
{
  "status": "success",
  "kernel_exec_result": {
    "compiled": true,
    "runtime": 0.123,
    "runtime_stats": {
      "mean": 0.123,
      "std": 0.005,
      "median": 0.122,
      "percentile_95": 0.130
    }
  }
}
```

## Core Concepts

### What's a Kernel?
A **kernel** is code that runs on the GPU. Think of it as a function optimized for parallel computation. We support multiple types:

- **PyTorch**: Standard deep learning models and operations
- **CUDA**: Low-level C++ code for maximum control
- **Triton**: Python-like language for writing GPU kernels
- **PyTorch+CUDA**: PyTorch models with embedded custom CUDA

### What Does the Server Do?

```
Your Code → [Server] → Performance Metrics
              ↓
        1. Compile
        2. Validate 
        3. Profile
        4. Analyze
```

The server:
1. **Receives** your kernel code via REST API
2. **Compiles** it for your specific GPU
3. **Validates** correctness (when comparing kernels)
4. **Profiles** performance over multiple trials
5. **Returns** detailed metrics and statistics

## API Overview

### Two Main Operations

#### 1. Evaluate Single Kernel (`/evaluate`)
**Use when**: Measuring standalone performance

```python
import requests

response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2",
        "kernel_type": "torch"
    },
    "num_trials": 100,
    "timeout": 120
})

result = response.json()
print(f"Runtime: {result['kernel_exec_result']['runtime']}ms")
```

#### 2. Compare Two Kernels (`/compare`)
**Use when**: A/B testing implementations

```python
response = requests.post("http://localhost:8000/compare", json={
    "ref_kernel": {
        "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return torch.matmul(x, x.T)",
        "kernel_type": "torch"
    },
    "custom_kernel": {
        "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return custom_matmul(x, x.T)",  # Your optimized version
        "kernel_type": "torch_cuda"
    },
    "num_trials": 200
})

result = response.json()
if result['kernel_exec_result']['correctness']:
    speedup = result['ref_runtime']['mean'] / result['kernel_exec_result']['runtime']
    print(f"Speedup: {speedup:.2f}x")
```

### Understanding Responses

```python
{
    "status": "success",              # Request status
    "kernel_exec_result": {
        "compiled": true,             # Did compilation succeed?
        "correctness": true,          # Does output match reference? (compare only)
        "runtime": 1.23,              # Mean runtime in milliseconds
        "runtime_stats": {            # Detailed statistics
            "mean": 1.23,
            "std": 0.05,
            "min": 1.18,
            "max": 1.35,
            "median": 1.22,
            "percentile_95": 1.30,
            "percentile_99": 1.33
        },
        "compilation_error": null,    # Error message if compilation failed
        "validation_error": null       # Error message if validation failed
    }
}
```

## Supported Kernel Types

| Type | Use Case | Example | IOContract Required |
|------|----------|---------|---------------------|
| `torch` | Standard ML models | ResNet, Transformers | No (auto-generates) |
| `torch_cuda` | PyTorch + custom CUDA | Optimized operators | No (uses get_inputs()) |
| `triton` | Custom GPU kernels | Matrix multiplication | **Yes** |
| `cuda` | Raw CUDA C++ | Low-level optimizations | **Yes** |

## Common Use Cases

### 1. Benchmark a PyTorch Model

```python
# Evaluate a transformer layer
response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(512, 8)
    
    def forward(self, x):
        return self.attention(x, x, x)[0]
""",
        "kernel_type": "torch"
    },
    "num_trials": 100
})
```

### 2. Compare Optimization Strategies

```python
# Compare standard vs flash attention
response = requests.post("http://localhost:8000/compare", json={
    "ref_kernel": {
        "source_code": "# Standard attention implementation",
        "kernel_type": "torch"
    },
    "custom_kernel": {
        "source_code": "# Flash attention implementation",
        "kernel_type": "torch_cuda"
    },
    "num_trials": 200
})
```

### 3. Profile a Triton Kernel

```python
# Triton kernel with explicit inputs
response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": """
import triton
import triton.language as tl

@triton.jit
def vector_add(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
""",
        "kernel_type": "triton",
        "io": {
            "args": [
                {"name": "x", "type": "tensor", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "randn", "seed": 42}}, "role": "input"},
                {"name": "y", "type": "tensor", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "randn", "seed": 43}}, "role": "input"},
                {"name": "output", "type": "tensor", "tensor_spec": {"shape": [1024], "dtype": "float32"}, "role": "output"},
                {"name": "n_elements", "type": "int", "value": 1024, "role": "input"},
                {"name": "BLOCK_SIZE", "type": "int", "value": 256, "role": "input", "is_meta": true}
            ],
            "launch": {"grid": {"x": 4}, "num_warps": 4}
        }
    },
    "num_trials": 100
})
```

For more examples and advanced usage, see [API_GUIDE.md](API_GUIDE.md).

## Additional Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Job Status
```bash
curl http://localhost:8000/job/{job_id}
```

### Server Statistics
```bash
curl http://localhost:8000/stats
```

### Admin: Cleanup Old Jobs
```bash
curl -X POST http://localhost:8000/admin/cleanup-jobs
```

## Troubleshooting

### Common Issues

| Problem | Symptom | Solution |
|---------|---------|----------|
| GPU not found | `"gpu_available": false` | Check CUDA installation with `nvidia-smi` |
| Compilation failed | `"compiled": false` | Check `compilation_error` field in response |
| Timeout | Request hangs | Increase `timeout` parameter (max: 600) |
| Port in use | Server won't start | Change port: `python main.py --port 8001` |
| Out of memory | CUDA OOM error | Reduce tensor sizes or batch size |
| Wrong results | `"correctness": false` | Check `validation_error`, verify algorithm |

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py

# Enable device metrics (NCU profiling)
ENABLE_DEVICE_METRICS=true python main.py

# View detailed logs
tail -f logs/cuda_eval_server.log
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPU_DEVICE` | GPU device ID to use | 0 |
| `MAX_WORKERS` | Parallel subprocess workers | 4 |
| `ENABLE_DEVICE_METRICS` | Enable NCU profiling | false |
| `LOG_LEVEL` | Logging verbosity | INFO |
| `PORT` | Server port | 8000 |
| `HOST` | Server host | 0.0.0.0 |

### Performance Tuning

- **Trial Count**: More trials = more accurate results (recommended: 100-1000)
- **Timeout**: Set based on kernel complexity (default: 120s, max: 600s)
- **Memory**: Server uses ~2GB base + kernel requirements
- **Overhead**: Typically 10-50ms per evaluation

## Advanced Features

### IOContract for Reproducible Testing
Specify exact inputs and outputs for deterministic testing. Required for Triton and CUDA kernels.

### Device Metrics with NCU
Enable detailed GPU profiling with NVIDIA Nsight Compute:
```bash
ENABLE_DEVICE_METRICS=true python main.py
```

### Function Targeting
Test specific functions within your code:
```python
"metadata": {
    "function_name": "optimized_softmax"  # Target specific function
}
```

### Multi-Kernel Support
CUDA files can contain multiple kernels with runtime selection.

For detailed documentation on these features, see [API_GUIDE.md](API_GUIDE.md).

## Architecture Overview

The server uses a modular architecture with pluggable backends:

```
Request → FastAPI → Job Manager → Subprocess Worker
                                    ↓
                            Compilation Backend
                            (Torch/CUDA/Triton)
                                    ↓
                            Validation & Profiling
                                    ↓
                                Response
```

Key design decisions:
- **Subprocess Isolation**: Prevents kernel crashes from affecting the server
- **Backend Pattern**: Easy to add support for new kernel types
- **Async Processing**: Non-blocking request handling
- **Statistical Profiling**: Multiple trials with percentile analysis

For implementation details, see [USER_MANUAL.md](USER_MANUAL.md).

## Development

### Project Structure

```
cuda_eval_server_v2/
├── app.py                    # FastAPI application
├── main.py                   # Entry point
├── subprocess_worker.py      # Isolated execution
├── orchestration/           # Job management
├── compilation/             # Kernel compilation backends
│   ├── torch/              # PyTorch backend
│   ├── torch_cuda/         # PyTorch+CUDA backend
│   ├── triton/             # Triton backend
│   └── cuda/               # Raw CUDA backend
├── validation/              # Correctness validation
├── profiling/               # Performance profiling
└── shared/                  # Common utilities
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_torch_backend.py

# Run with coverage
pytest --cov=. tests/
```

### Adding New Kernel Types

1. Create a new backend in `compilation/`
2. Inherit from `BaseCompilationBackend`
3. Implement `compile()` method
4. Register in `CompilationService`

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Related Documentation

- [API_GUIDE.md](API_GUIDE.md) - Complete API reference with examples
- [USER_MANUAL.md](USER_MANUAL.md) - System design and implementation details
- [DEVICE_METRICS_GUIDE.md](DEVICE_METRICS_GUIDE.md) - GPU profiling with NCU
- [examples/](examples/) - Sample kernels and use cases
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

## License

MIT License - see [LICENSE](LICENSE) for details.