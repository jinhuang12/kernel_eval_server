# Kernel Evaluation Client Library

Python client library for interacting with the Kernel Evaluation Server API. Simplifies building IOContracts and submitting kernel evaluation requests.

## Features

- 🚀 **Simple API** - Fluent interface for building complex IOContracts
- 🎯 **Type Safety** - Python dataclasses with validation
- 📦 **Minimal Dependencies** - Only requires `requests` library
- 🔧 **All Kernel Types** - Support for PyTorch, CUDA, Triton, and PyTorch+CUDA
- 📊 **Response Parsing** - Automatic parsing of performance metrics
- 💾 **JSON Serialization** - Save/load IOContracts to/from files

## Architecture Note

This client library is part of the KernelBench repository and shares core data models with the server implementation. The models are imported from `shared.models` to ensure consistency between client and server. Client-specific models (like `RuntimeStats` for response parsing) are defined in the client package.

## Installation

### Local Development

```bash
cd KernelBench/scripts/cuda_eval_server_v2/client
pip install -e .
```

**Note**: Since the client imports from the server's `shared` module, it must be installed from within the KernelBench repository structure.

### Install with Examples

```bash
pip install -e ".[examples]"  # Includes numpy for advanced examples
```

## Quick Start

### Basic Usage

```python
from kernel_eval_client import KernelEvalClient, KernelCode, KernelType

# Create client
client = KernelEvalClient("http://localhost:8000")

# Define a kernel
kernel = KernelCode(
    source_code="import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return torch.relu(x)",
    kernel_type=KernelType.TORCH
)

# Evaluate kernel
result = client.evaluate(kernel, num_trials=100)
print(f"Runtime: {result['kernel_exec_result']['runtime']:.3f} ms")
```

### Building IOContracts

```python
from kernel_eval_client import IOContractBuilder, create_randn_spec

# Build IOContract with fluent API
io_contract = (
    IOContractBuilder()
    .add_input_tensor("x", create_randn_spec([1024, 512], seed=42))
    .add_input_tensor("y", create_randn_spec([512, 256], seed=43))
    .add_output_tensor("result", [1024, 256], "float32")
    .add_scalar("alpha", "float", 2.0)
    .set_grid(16, 16)  # CUDA/Triton grid dimensions
    .set_block(16, 16)  # CUDA block dimensions
    .build()
)
```

### Triton Kernel Example

```python
from kernel_eval_client import KernelCode, KernelType, IOContractBuilder

triton_source = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
"""

# Build IOContract for Triton
io_contract = (
    IOContractBuilder()
    .add_input_tensor("x", create_randn_spec([1024]))
    .add_input_tensor("y", create_randn_spec([1024]))
    .add_output_tensor("output", [1024])
    .add_scalar("n_elements", "int", 1024)
    .add_meta_param("BLOCK_SIZE", 256)  # Triton constexpr
    .set_grid(4)
    .set_num_warps(4)
    .build()
)

kernel = KernelCode(source_code=triton_source, kernel_type=KernelType.TRITON, io=io_contract)
result = client.evaluate(kernel)
```

### CUDA Kernel Example

```python
cuda_source = """
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}
"""

io_contract = (
    IOContractBuilder()
    .add_input_tensor("a", create_randn_spec([1024]))
    .add_input_tensor("b", create_randn_spec([1024]))
    .add_output_tensor("c", [1024])
    .add_scalar("n", "int", 1024)
    .set_grid(4)
    .set_block(256)
    .build()
)

kernel = KernelCode(source_code=cuda_source, kernel_type=KernelType.CUDA, io=io_contract)
```

## Tensor Initialization

### Server-side Generation

```python
from kernel_eval_client import create_randn_spec, create_uniform_spec, create_zeros_spec

# Random normal distribution
randn_spec = create_randn_spec([512, 512], seed=42, mean=0.0, std=1.0)

# Uniform distribution
uniform_spec = create_uniform_spec([512, 512], seed=43, low=-1.0, high=1.0)

# Zeros/Ones
zeros_spec = create_zeros_spec([512, 512])
ones_spec = create_ones_spec([512, 512])

# Constant value
full_spec = create_full_spec([512, 512], fill_value=3.14)

# Sequential values
arange_spec = create_arange_spec([1024], start=0.0, step=0.1)
```

### Custom Tensor Data

```python
import numpy as np
import base64
from kernel_eval_client import TensorSpec, TensorData

# Create custom data
data = np.random.randn(512, 512).astype(np.float32)
data_b64 = base64.b64encode(data.tobytes()).decode('utf-8')

# Create TensorSpec with literal data
tensor_spec = TensorSpec(
    shape=[512, 512],
    dtype="float32",
    data=TensorData(
        data_b64=data_b64,
        dtype="float32",
        shape=[512, 512]
    )
)
```

## PyTorch Function Targeting

Target specific functions or methods in PyTorch code:

```python
source_code = """
import torch

def my_function(x, y):
    return x + y

class MyModel(torch.nn.Module):
    def special_method(self, x):
        return torch.relu(x)
"""

# Target function
kernel = KernelCode(
    source_code=source_code,
    kernel_type=KernelType.TORCH,
    io=io_contract,
    metadata={"function_name": "my_function"}
)

# Target class method
kernel = KernelCode(
    source_code=source_code,
    kernel_type=KernelType.TORCH,
    io=io_contract,
    metadata={
        "class_name": "MyModel",
        "method_name": "special_method"
    }
)
```

## JSON Serialization

```python
from kernel_eval_client import save_to_file, load_from_file, to_json

# Save IOContract to file
save_to_file(io_contract, "my_contract.json")

# Load IOContract from file
loaded_contract = load_from_file("my_contract.json", "IOContract")

# Convert to JSON string
json_str = to_json(io_contract, pretty=True)
```

## API Reference

### KernelEvalClient

- `compare(ref_kernel, custom_kernel, num_trials=100)` - Compare two kernels
- `evaluate(kernel, num_trials=100)` - Evaluate single kernel
- `health_check()` - Check server health
- `get_job_status(job_id)` - Get job status
- `get_stats()` - Get server statistics

### IOContractBuilder

- `add_input_tensor(name, tensor_spec)` - Add input tensor
- `add_output_tensor(name, shape, dtype)` - Add output tensor
- `add_inout_tensor(name, tensor_spec)` - Add input/output tensor
- `add_scalar(name, type, value)` - Add scalar argument
- `add_meta_param(name, value)` - Add Triton constexpr
- `set_grid(x, y, z)` - Set grid dimensions
- `set_block(x, y, z)` - Set block dimensions (CUDA)
- `set_num_warps(n)` - Set warps (Triton)
- `build()` - Build final IOContract

## Examples

See the `examples/` directory for complete examples:

- `simple_pytorch.py` - Basic PyTorch kernel evaluation
- `triton_kernel.py` - Triton kernel with IOContract
- `cuda_kernel.py` - CUDA kernels with launch configuration
- `advanced_patterns.py` - Advanced usage patterns
