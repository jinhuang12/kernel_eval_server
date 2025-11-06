# Kernel Evaluation Server API Guide

## 1. Getting Started (< 1 minute to first API call)

### Simplest Example - Evaluate a PyTorch Kernel

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "kernel": {
      "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2",
      "kernel_type": "torch",
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
    },
    "num_trials": 100
  }'
```

### Compare Two Kernels

```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "ref_kernel": {
      "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2",
      "kernel_type": "torch",
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
    },
    "custom_kernel": {
      "source_code": "class Model(torch.nn.Module):\n    def forward(self, x):\n        return x + x",
      "kernel_type": "torch",
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
    },
    "num_trials": 100
  }'
```

### Using Python Client

```bash
# Go into directory where the package is checked out
cd workspace/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2/client
pip install -e .
```

```python
from kernel_eval_client import KernelEvalClient, KernelCode, KernelType, IOContractBuilder, create_randn_spec

client = KernelEvalClient("http://localhost:8000")

# Define input args for kernel
inputs = IOContractBuilder().add_input_tensor("x", create_randn_spec([1024, 512], seed=42)).build()

# Evaluate a single kernel
kernel = KernelCode(
    source_code="class Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2",
    kernel_type=KernelType.TORCH,
    io=inputs
)
result = client.evaluate(kernel, num_trials=100)
print(f"Runtime: {result.kernel_exec_result.runtime}ms")
```

## 2. Which Endpoint Should I Use?

### Decision Tree
```
┌─────────────────────────────────────┐
│  What do you want to do?            │
└─────────────┬───────────────────────┘
              │
              ├─ Measure ONE kernel's performance ──→ Use `/evaluate`
              │
              ├─ Compare TWO kernels ──────────────→ Use `/compare`
              │
              ├─ Check server health ───────────────→ Use `/health`
              │
              ├─ Get job status ────────────────────→ Use `/job/{job_id}`
              │
              └─ View server statistics ────────────→ Use `/stats`
```

### Endpoint Summary

| Endpoint | Purpose | When to Use | Response Time |
|----------|---------|-------------|---------------|
| **POST** `/evaluate` | Single kernel profiling | Testing standalone performance | 1-30s |
| **POST** `/compare` | Compare two kernels | A/B testing implementations | 2-60s |
| **GET** `/health` | Server status | Monitoring/debugging | <100ms |
| **GET** `/job/{id}` | Job status | Async job tracking | <100ms |
| **GET** `/stats` | Server metrics | Performance monitoring | <100ms |

## 3. Kernel Types Quick Reference

### What kernel_type should I use?

| Your Code | kernel_type | IOContract Required? | Example Use Case |
|-----------|-------------|---------------------|------------------|
| Pure PyTorch | `"torch"` | No (auto-generates) | Reference implementations |
| PyTorch + CUDA | `"torch_cuda"` | No (uses KernelBench format) | Optimized PyTorch models |
| Triton kernel | `"triton"` | **Yes** | Custom GPU kernels |
| Raw CUDA | `"cuda"` | **Yes** | Low-level GPU code |

## 4. Common Use Cases

### 4.1 Evaluate a PyTorch Model

```python
import requests

# Simple PyTorch model evaluation - server auto-generates inputs
response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": """
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256)
    
    def forward(self, x):
        return torch.relu(self.linear(x))
""",
        "kernel_type": "torch",
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
    },
    "num_trials": 100
})

result = response.json()
print(f"Compiled: {result['kernel_exec_result']['compiled']}")
print(f"Runtime: {result['kernel_exec_result']['runtime']}ms")
```

### 4.2 Compare PyTorch vs Optimized Implementation

```python
# Compare reference PyTorch against optimized version
response = requests.post("http://localhost:8000/compare", json={
    "ref_kernel": {
        "source_code": """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
""",
        "kernel_type": "torch"
    },
    "custom_kernel": {
        "source_code": """
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
\"\"\"

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
""",
        "kernel_type": "torch_cuda"
    },
    "num_trials": 200
})

result = response.json()
ref_time = result['ref_runtime']['mean']
custom_time = result['kernel_exec_result']['runtime']
print(f"Speedup: {ref_time / custom_time:.2f}x")
```

### 4.3 Triton Kernel with Custom Inputs

```python
# Triton kernel requires IOContract for input specification
response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
""",
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
                    "name": "y",
                    "type": "tensor",
                    "tensor_spec": {
                        "shape": [1024],
                        "dtype": "float32",
                        "init": {"kind": "ones"}
                    },
                    "role": "input"
                },
                {
                    "name": "output",
                    "type": "tensor",
                    "tensor_spec": {
                        "shape": [1024],
                        "dtype": "float32"
                    },
                    "role": "output"
                },
                {
                    "name": "n_elements",
                    "type": "int",
                    "value": 1024,
                    "role": "input"
                },
                {
                    "name": "BLOCK_SIZE",
                    "type": "int",
                    "value": 256,
                    "role": "input",
                    "is_meta": True  # Triton compile-time constant
                }
            ],
            "launch": {
                "grid": {"x": 4},  # 1024 / 256 = 4 blocks
                "num_warps": 4
            }
        }
    },
    "num_trials": 100
})
```

### 4.4 CUDA Kernel Evaluation

```python
# Raw CUDA kernel with launch configuration
response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
""",
        "kernel_type": "cuda",
        "metadata": {
            "kernel_name": "vector_add",  # Required for CUDA
            "compiler_options": ["--use_fast_math"]
        },
        "io": {
            "args": [
                {"name": "a", "type": "tensor", "tensor_spec": {"shape": [4096], "dtype": "float32", "init": {"kind": "randn", "seed": 1}}, "role": "input"},
                {"name": "b", "type": "tensor", "tensor_spec": {"shape": [4096], "dtype": "float32", "init": {"kind": "randn", "seed": 2}}, "role": "input"},
                {"name": "c", "type": "tensor", "tensor_spec": {"shape": [4096], "dtype": "float32"}, "role": "output"},
                {"name": "n", "type": "int", "value": 4096, "role": "input"}
            ],
            "launch": {
                "grid": {"x": 16},
                "block": {"x": 256}
            }
        }
    },
    "num_trials": 100
})
```

### 4.5 Target Specific PyTorch Functions

```python
# Target a specific function instead of Model.forward()
response = requests.post("http://localhost:8000/evaluate", json={
    "kernel": {
        "source_code": """
import torch

def optimized_softmax(x):
    return torch.softmax(x, dim=-1)

def another_function(x):
    return x * 2  # This won't be called

class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1  # This won't be called either
""",
        "kernel_type": "torch",
        "metadata": {
            "function_name": "optimized_softmax"  # Target specific function
        },
        "io": {
            "args": [
                {
                    "name": "x",
                    "type": "tensor",
                    "tensor_spec": {
                        "shape": [64, 128],
                        "dtype": "float32",
                        "init": {"kind": "randn", "seed": 42}
                    }
                }
            ]
        }
    },
    "num_trials": 100
})
```

## 5. Understanding Responses

### Success Response Structure

```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "success",
    "kernel_exec_result": {
        "compiled": true,           // Did compilation succeed?
        "correctness": true,        // Is output correct? (comparison only)
        "runtime": 1.23,            // Mean runtime in milliseconds
        "runtime_stats": {          // Detailed performance metrics
            "mean": 1.23,
            "std": 0.05,
            "min": 1.18,
            "max": 1.35,
            "median": 1.22,
            "percentile_95": 1.30,
            "percentile_99": 1.33
        },
        "metadata": {               // GPU and kernel information
            "gpu_id": 0,
            "gpu_type": "NVIDIA H100",
            "device_metrics": {
                // Kernel device metrics (/evaluate) | Reference kernel device metrics (/compare)
                "original_device_metrics": {
                    "speed_of_light": {
                        "compute_throughput_pct": 85.2,
                        "memory_throughput_pct": 92.1,
                        "compute_memory_throughput_pct": 88.5,
                        "sm_throughput_pct": 82.3,
                        "dram_throughput_pct": 91.8,
                        "gpu_dram_throughput_pct": 90.2
                    },
                    "detailed_metrics": {
                        "l1_hit_rate_pct": 89.2,
                        "l2_hit_rate_pct": 73.4,
                        "warp_occupancy_pct": 78.9,
                        "sm_active_cycles_pct": 92.1,
                        "instructions_per_cycle": 2.34,
                        "waves_per_sm": 3.2
                    },
                    "memory_metrics": {
                        "dram_avg_bandwidth_gb_s": 456.2,
                        "dram_total_bandwidth_gb_s": 512.8,
                        "dram_active_cycles_pct": 88.4,
                        "l1_writeback_active_pct": 45.2,
                        "l1_read_sectors_pct": 67.8,
                        "l2_throughput_pct": 82.3
                    },
                    "compute_metrics": {
                        "fma_pipe_utilization_pct": 75.2,
                        "fp64_pipe_utilization_pct": 12.3,
                        "alu_pipe_utilization_pct": 68.9,
                        "xu_pipe_utilization_pct": 34.5,
                        "tensor_pipe_utilization_pct": 0.0,
                        "instructions_per_cycle": 2.34,
                        "occupancy_limit_blocks": 16,
                        "occupancy_limit_registers": 65536,
                        "occupancy_limit_shared_mem": 49152,
                        "occupancy_limit_warps": 64,
                        "registers_per_thread": 32
                    },
                    "pipeline_metrics": {
                        "fma_pipe_active_pct": 72.3,
                        "alu_pipe_active_pct": 65.4,
                        "tensor_pipe_active_pct": 0.0,
                        "shared_pipe_active_pct": 45.6,
                        "fp64_pipe_active_pct": 10.2,
                        "sm_issue_active_pct": 88.9
                    },
                    "occupancy_metrics": {
                        "occupancy_limit_registers": 65536,
                        "occupancy_limit_shared_mem": 49152,
                        "occupancy_limit_warps": 64,
                        "occupancy_limit_blocks": 16,
                        "waves_per_sm": 3.2,
                        "block_size": 256,
                        "grid_size": 1024,
                        "shared_mem_per_block": 4096
                    },
                    "stall_metrics": {
                        "stall_long_scoreboard_pct": 28.4,
                        "stall_short_scoreboard_pct": 6.1,
                        "stall_barrier_pct": 2.7,
                        "stall_not_selected_pct": 14.9
                    },
                    "scheduler_metrics": {
                        "warps_eligible_per_cycle": 5.6,
                        "inst_issued_per_cycle": 2.9,
                        "issue_active_pct": 71.2
                    },
                    "access_pattern_metrics": {
                        "l1_load_sectors_per_req": 1.85,
                        "l1_store_sectors_per_req": 2.10,
                        "l2_theoretical_sectors_global": 1.23e8,
                        "l2_theoretical_sectors_global_ideal": 9.10e7,
                        "l2_theoretical_sectors_global_excessive": 3.20e7,
                        "l2_excess_frac": 0.26,
                        "shared_bank_conflicts_load_sum": 4096,
                        "shared_bank_conflicts_store_sum": 0
                    },
                    "roofline_metrics": {
                        "flop_count_total": 2.10e12,
                        "dram_bytes_sum": 5.80e11,
                        "arithmetic_intensity": 3.62,
                        "gflops": 14500.0
                    },
                    "timing_metrics": {
                        "gpu_time_duration_sum": 14.5,
                        "gpu_time_duration_unit": "ms",
                        "gpu_time_duration_seconds": 0.0145,
                        "gpc_cycles_elapsed_max": 3.1e8
                    }
                }
            }    
        }
    }
}
```

### Common Response Patterns

#### ✅ Successful Evaluation
```json
{
    "status": "success",
    "kernel_exec_result": {
        "compiled": true,
        "runtime": 1.23,
        "compilation_error": null
    }
}
```

#### ⚠️ Compilation Failed
```json
{
    "status": "success",  // Note: Still returns success
    "kernel_exec_result": {
        "compiled": false,
        "runtime": 0,
        "compilation_error": "Syntax error at line 10: unexpected token"
    }
}
```

#### ❌ Validation Failed (Comparison Only)
```json
{
    "status": "success",
    "kernel_exec_result": {
        "compiled": true,
        "correctness": false,
        "runtime": 0,  // Profiling skipped
        "validation_error": "Maximum difference: 0.5 exceeds tolerance 1e-4"
    }
}
```

## 6. IOContract - When and How to Use It

### When Do I Need IOContract?

| Kernel Type | IOContract Required? | What Happens Without It? |
|-------------|---------------------|-------------------------|
| `torch` | Optional | Auto-generates random inputs via `get_inputs()` |
| `torch_cuda` | Optional | Uses KernelBench format (`get_inputs()` function) |
| `triton` | **Required** | Error - cannot infer inputs |
| `cuda` | **Required** | Error - cannot infer inputs |

### Simple IOContract Example

```python
# Minimal IOContract for a vector addition kernel
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
            "name": "y",
            "type": "tensor",
            "tensor_spec": {
                "shape": [1024],
                "dtype": "float32",
                "init": {"kind": "randn", "seed": 43}
            },
            "role": "input"
        }
    ]
}
```

### IOContract with Mixed Types

```python
"io": {
    "args": [
        # Tensor input
        {
            "name": "data",
            "type": "tensor",
            "tensor_spec": {
                "shape": [512, 768],
                "dtype": "float32",
                "init": {"kind": "randn", "seed": 42}
            },
            "role": "input"
        },
        # Scalar parameter
        {
            "name": "scale",
            "type": "float",
            "value": 2.0,
            "role": "input"
        },
        # Integer size
        {
            "name": "size",
            "type": "int",
            "value": 512,
            "role": "input"
        },
        # Output tensor (shape only, no init)
        {
            "name": "output",
            "type": "tensor",
            "tensor_spec": {
                "shape": [512, 768],
                "dtype": "float32"
            },
            "role": "output"
        }
    ]
}
```

## 7. Advanced Features

### 7.1 Tensor Generation Methods

Generate tensors server-side for reproducible testing:

| Method | Parameters | Use Case |
|--------|-----------|----------|
| `"randn"` | `mean`, `std`, `seed` | Normal distribution |
| `"uniform"` | `low`, `high`, `seed` | Uniform distribution |
| `"zeros"` | `seed` | Zero tensor |
| `"ones"` | `seed` | Ones tensor |
| `"full"` | `fill_value`, `seed` | Constant value |
| `"arange"` | `start`, `step` | Sequential values |

Example:
```python
"tensor_spec": {
    "shape": [1024, 512],
    "dtype": "float32",
    "init": {
        "kind": "uniform",
        "low": -1.0,
        "high": 1.0,
        "seed": 42
    }
}
```

### 7.2 Launch Configuration

#### For CUDA Kernels
```python
"launch": {
    "grid": {"x": 32, "y": 1, "z": 1},   # Grid dimensions
    "block": {"x": 256, "y": 1, "z": 1}  # Block dimensions
}
```

#### For Triton Kernels
```python
"launch": {
    "grid": {"x": 32},      # Grid size
    "num_warps": 4,         # Warps per block (default: 4)
    "num_stages": 3         # Pipeline stages (default: 3)
}
```

### 7.3 Kernel Metadata

#### PyTorch Metadata
```python
"metadata": {
    "function_name": str,    # Target standalone function
    "class_name": str,       # Target class (default: "Model")
    "method_name": str       # Target method (default: "forward")
}
```

#### CUDA Metadata
```python
"metadata": {
    "kernel_name": str,           # Entrypoint kernel name (required)
    "compiler_options": [str],    # NVRTC/NVCC flags
    "backend": str,              # "nvrtc" (default) or "nvcc"
    "jitify": bool               # Enable C++ features (default: false)
}
```

#### Triton Metadata
```python
"metadata": {
    "kernel_name": str,              # Entry point function (for multi-kernel files)
}
```

### 7.4 Device Metrics (NCU Profiling)

When NCU profiling is enabled, responses include detailed GPU metrics:

```json
"device_metrics": {
    "speed_of_light": {
        "compute_throughput_pct": 60.4,
        "memory_throughput_pct": 35.7
    },
    "detailed_metrics": {
        "l1_hit_rate_pct": 95.2,
        "l2_hit_rate_pct": 88.7,
        "warp_occupancy_pct": 75.0,
        "achieved_occupancy_pct": 68.3
    },
    "memory_metrics": {
        "dram_throughput_gbps": 1200.5,
        "l2_throughput_gbps": 3500.2
    }
}
```

### 7.5 Multi-Kernel CUDA Files

For CUDA files with multiple kernels:

```python
"kernel": {
    "source_code": """
__global__ void kernel1(...) { ... }
__global__ void kernel2(...) { ... }
__global__ void kernel3(...) { ... }
""",
    "kernel_type": "cuda",
    "metadata": {
        "kernel_name": "kernel2"  # Specify which kernel to run
    }
}
```

## 8. Troubleshooting

### Common Issues and Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| Kernel doesn't compile | `"compiled": false` | Check `compilation_error` field for details |
| Results don't match | `"correctness": false` | Check `validation_error`, adjust tolerance |
| Timeout errors | HTTP 408 or timeout message | Increase `timeout` parameter (max: 600) |
| IOContract confusion | "Missing required args" | Use client library helpers or examples |
| Server unavailable | Connection refused | Check server is running with `/health` |
| GPU OOM | "CUDA out of memory" | Reduce batch size or tensor dimensions |
| Wrong kernel called | Unexpected results | Check `metadata.kernel_name` for CUDA/Triton |

### Debugging Tips

1. **Start simple**: Test with minimal code first
2. **Check compilation**: Ensure `compiled: true` before debugging logic
3. **Use single evaluation**: Debug with `/evaluate` before `/compare`
4. **Verify inputs**: Print tensor shapes/values in kernel code
5. **Check GPU**: Use `/stats` to verify GPU availability

### Error Message Guide

```python
# Compilation error
"compilation_error": "Syntax error at line 10: unexpected token"
# → Fix syntax in source_code

# Validation error  
"validation_error": "Maximum difference: 0.5 exceeds tolerance"
# → Results don't match reference, check algorithm

# Timeout error
"error": "Evaluation timed out after 120 seconds"
# → Increase timeout or optimize kernel

# IOContract error
"error": "Missing required argument: BLOCK_SIZE"
# → Add missing argument to io.args
```

## 9. API Reference

### Request Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `kernel` or `ref_kernel` | KernelCode | Yes | - | Kernel to evaluate |
| `custom_kernel` | KernelCode | For `/compare` | - | Second kernel for comparison |
| `num_trials` | int | No | 100 | Number of profiling iterations |
| `timeout` | int | No | 120 | Timeout in seconds (max: 600) |

### KernelCode Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_code` | string | Yes | Kernel source code |
| `kernel_type` | string | Yes | One of: "torch", "torch_cuda", "triton", "cuda" |
| `io` | IOContract | Depends | Required for Triton/CUDA |
| `metadata` | dict | No | Kernel-specific metadata |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `status` | string | "success" or "error" |
| `kernel_exec_result` | object | Execution results |
| `ref_runtime` | object | Reference kernel stats (comparison only) |

## 10. Client Libraries

### Python Client Installation

```bash
# Go into directory where the package is checked out
cd workspace/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2/client
pip install -e .
```

### Python Client Examples

#### Basic Usage
```python
from kernel_eval_client import KernelEvalClient, KernelCode, KernelType

client = KernelEvalClient("http://localhost:8000")

# Single evaluation
result = client.evaluate(
    KernelCode(source_code="...", kernel_type=KernelType.TORCH),
    num_trials=100
)

# Comparison
result = client.compare(
    ref_kernel=KernelCode(...),
    custom_kernel=KernelCode(...),
    num_trials=100
)
```

#### Building IOContracts
```python
from kernel_eval_client import IOContractBuilder, create_randn_spec

io = (
    IOContractBuilder()
    .add_input_tensor("x", create_randn_spec([1024, 512], seed=42))
    .add_output_tensor("y", [1024, 512], "float32")
    .add_scalar("alpha", "float", 2.0)
    .set_grid(32)
    .set_block(256)
    .build()
)
```

## Appendix A: Legacy Format Support

For backward compatibility, the server still accepts the old format:

```python
# Old format (still works but not recommended)
{
    "ref_code": "class Model...",
    "custom_code": "class Model...",
    "num_trials": 100
}

# Automatically converted to:
{
    "ref_kernel": {"source_code": "...", "kernel_type": "torch"},
    "custom_kernel": {"source_code": "...", "kernel_type": "torch_cuda"}
}
```

**Note**: Use the new format for better control and clarity.

## Appendix B: Performance Best Practices

1. **More trials = better accuracy**: Use 100+ trials for stable measurements
2. **Warmup is automatic**: Server handles GPU warmup
3. **CUDA Graphs**: Automatically enabled for better profiling
4. **Subprocess isolation**: Each eval runs in subprocess for safety
5. **Deterministic seeds**: Use same seeds for reproducible results
6. **Timeout appropriately**: Set based on kernel complexity
7. **Profile incrementally**: Start with small inputs, scale up

## Appendix C: Server Configuration

### Environment Variables
- `GPU_DEVICE`: GPU to use (default: 0)
- `ENABLE_NCU`: Enable NCU profiling (default: false)

### Health Check Response
```json
{
    "status": "healthy",
    "compilation_service": "ready",
    "profiling_service": "ready",
    "gpu_available": true,
    "gpu_count": 8
}
```

### Server Statistics
```json
{
    "jobs_completed": 1234,
    "jobs_failed": 5,
    "avg_runtime_ms": 1500,
    "gpu_utilization_pct": 45.2,
    "uptime_seconds": 3600
}
```