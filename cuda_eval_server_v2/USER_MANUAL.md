# CUDA Evaluation Server V2 - Comprehensive User Manual

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Kernel Type Support](#kernel-type-support)
3. [Source Code Constraints & Guidelines](#source-code-constraints--guidelines)
4. [Compilation Process](#compilation-process)
5. [Validation Mechanisms](#validation-mechanisms)
6. [Profiling Implementation](#profiling-implementation)
7. [IOContract Specification](#iocontract-specification)
8. [API Endpoints](#api-endpoints)
9. [Implementation Details](#implementation-details)
10. [Best Practices](#best-practices)

## System Architecture

### Request Flow Pipeline
```
Client Request → FastAPI (app.py) → JobManager → GPU Acquisition → Subprocess Worker → Result
```

### Component Isolation
The server employs **subprocess isolation** to prevent kernel crashes from affecting the main server:

1. **Main Process (app.py + JobManager)**
   - Handles HTTP requests asynchronously
   - Manages GPU resource allocation
   - Creates subprocess workers for actual execution
   - Maintains job state persistence

2. **Subprocess Worker (subprocess_worker.py)**
   - Executes compilation, validation, and profiling
   - Runs in complete isolation
   - Saves intermediate state to `/tmp/job_{job_id}_result.json`
   - Returns results even on partial failure

### GPU Resource Management
- **GPUResourceManager**: Tracks available GPUs using asyncio locks
- **GPU Acquisition**: Context manager pattern ensures proper release
- **Device Detection**: Automatic GPU type detection (A100/H100/H200)

### Job State Management
```python
@dataclass
class JobState:
    job_id: str
    status: str  # submitted, compiling, validating, profiling, completed, failed
    compilation_result: Optional[CompilationResult]
    validation_result: Optional[ValidationResult]
    profiling_result: Optional[ProfilingResult]
    result: Optional[Response]
```

## Kernel Type Support

**Overview**

Given the source code of a kernel & invocation argments, the eval server attempts to compile, execute & profile the kernel on GPU.
You can import multiple functions in single source code string, but can only target one of the functions. Your invocation args will be used as
input to the targeted function.

**Supported Kernel Types**:

| Type | Use Case | IOContract | Auto-generates Inputs |
|------|----------|------------|----------------------|
| `torch` | Pure PyTorch reference models | Optional | Yes (from model) |
| `torch_cuda` | PyTorch + embedded CUDA | Optional | Yes (from `get_inputs()`) |
| `triton` | Triton kernels | **Required** | No |
| `cuda` | Raw CUDA C++ | **Required** | No |
| `multi_kernel` | Python scripts with mixed kernel types | **Required** | No |

### 1. TORCH (Pure PyTorch)

**Compilation**:
- Direct module loading via `TorchCompilationBackend`
- Dynamic import and namespace execution
- Flexible targeting via `TorchKernelMetadata`

**Function Targeting via TorchKernelMetadata**:

The server uses `TorchKernelMetadata` to precisely target which function or method to execute:

```python
# Option 1: Target a standalone function
metadata = {
    "function_name": "my_kernel"  # Executes the standalone function
}

# Option 2: Target a class method
metadata = {
    "class_name": "MyModel",      # Class to instantiate
    "method_name": "forward"       # Method to call on instance
}
```

**Important**: Function targeting is **mutually exclusive**:
- Use `function_name` for standalone functions OR
- Use `class_name` + `method_name` for class methods
- Cannot specify both - the server will prioritize `function_name` if present

**Default Behavior** (when no metadata provided):

1. **With IOContract + Metadata**: Uses targeted execution based on metadata
2. **With IOContract only**: Attempts to create simple model wrapper, looks for `Model` class
3. **Fallback (KernelBench pattern)**: Searches for:
   - `Model` class with `forward()` method
   - `get_inputs()` function for generating inputs
   - `get_init_inputs()` function for model initialization
4. **Default values**:
   - Default class name: `"Model"` (if not specified in metadata)
   - Default method name: `"forward"` (if not specified in metadata)

**Example with Metadata**:
```python
# Source code with multiple execution targets
import torch
import torch.nn as nn

def standalone_kernel(x, y):
    """Standalone function that can be targeted"""
    return torch.matmul(x, y) + torch.relu(x)

class CustomModel(nn.Module):
    """Class with methods that can be targeted"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 128)
    
    def forward(self, x):
        return self.linear(x)
    
    def custom_method(self, x, y):
        return x + y

# To execute standalone_kernel:
# metadata = {"function_name": "standalone_kernel"}

# To execute CustomModel.forward:
# metadata = {"class_name": "CustomModel", "method_name": "forward"}

# To execute CustomModel.custom_method:
# metadata = {"class_name": "CustomModel", "method_name": "custom_method"}
```

### 2. TORCH_CUDA (PyTorch with Embedded CUDA)

**Purpose & Context**: 
- **Primary Use Case**: Designed specifically to support KernelBench generated code patterns
- **Implementation**: Rigid extraction pattern based on AST parsing - source code MUST follow exact structure
- **Not General Purpose**: This kernel type has strict requirements due to its specialized extraction process

**Critical Source Code Structure Requirements**:

The TORCH_CUDA extractor (`TorchCudaExtractor`) uses **regex-based parsing** that requires exact patterns:

1. **Variable Definition Order**:
   ```python
   # MUST define variables BEFORE load_inline call
   cuda_source = '''...'''  # Variable name extracted by regex
   cpp_source = '''...'''    # Variable name extracted by regex
   
   # THEN call load_inline with those variable names
   module = load_inline(
       name='kernel_name',
       cuda_sources=cuda_source,  # Must reference the variable
       cpp_sources=cpp_source,     # Must reference the variable
       functions=['forward']       # C++ functions to expose
   )
   ```

2. **Exact Parameter Names in load_inline**:
   - MUST use `cuda_sources` (not `cuda_source` or other variations)
   - MUST use `cpp_sources` (not `cpp_source` or other variations)
   - MUST include `functions` parameter with list of C++ function names

3. **ModelNew Class Requirement**:
   ```python
   # MUST define ModelNew class (NOT Model)
   class ModelNew(nn.Module):
       def forward(self, x):
           return module.forward(x)  # Uses the load_inline module
   ```

**Extraction Process Details**:

1. **Variable Name Extraction**:
   - Regex pattern searches for `load_inline\s*\((.*?)\)`
   - Extracts variable names from parameters: `cuda_sources\s*=\s*([^,\)]+)`
   - Looks up those variable definitions in source

2. **CUDA Source Extraction**:
   - Searches for triple-quoted blocks containing CUDA code
   - Identifies `__global__` kernel functions
   - Extracts includes and kernel body

3. **C++ Wrapper Extraction**:
   - Finds C++ function implementations
   - May extract from variable assignment or triple-quoted blocks
   - Matches function names with `functions` parameter

4. **ModelNew Extraction**:
   - Specifically searches for `class ModelNew`
   - Extracts complete class definition with proper indentation
   - Filters out compilation code and intermediate variables

**C++ Wrapper Transformation Process**:

1. **Function Signature Modification**:
   ```cpp
   // Original signature
   torch::Tensor forward(torch::Tensor x, torch::Tensor y);
   
   // Transformed signature (adds func_ptr)
   torch::Tensor forward(uint64_t func_ptr, torch::Tensor x, torch::Tensor y);
   ```

2. **Data Pointer Extraction**:
   ```cpp
   // Transforms tensor.data_ptr calls
   float* x_ptr = x.data_ptr<float>();
   float* y_ptr = y.data_ptr<float>();
   ```

3. **Kernel Launch Transformation**:
   ```cpp
   // Original kernel launch
   my_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>());
   
   // Transformed to use CuPy function pointer
   typedef void (*kernel_fn)(dim3, dim3, void**, size_t, cudaStream_t);
   void* args[] = {&x_ptr, &y_ptr};
   ((kernel_fn)func_ptr)(grid, block, args, 0, stream);
   ```

**Common Extraction Failures**:

- **"No load_inline call found"**: Pattern doesn't match expected format
- **"No cuda_sources found"**: Parameter name mismatch or missing
- **"Failed to extract CUDA source from variable"**: Variable not defined or wrong name
- **"No ModelNew torch module found"**: Must use ModelNew, not Model
- **"No __global__ functions found"**: CUDA kernel must have `__global__` qualifier

**Compilation Pipeline**:
1. **Extraction** (`TorchCudaExtractor`):
   - Regex-based pattern matching
   - Extracts CUDA kernel, C++ wrapper, and ModelNew class

2. **CuPy Compilation** (`TorchCudaCupyCompiler`):
   - Preprocesses CUDA source with `extern "C"`
   - Compiles with `cupy.RawKernel`
   - Creates function pointer registry

3. **C++ Wrapper Transformation** (`CppWrapperTransformer`):
   - Regex-based C++ transformation
   - Adds `func_ptr` parameter
   - Replaces kernel launches with CUDA Driver API calls

4. **Execution** (`TorchCudaExecutableKernel`):
   - Injects CuPy kernel pointer into transformed C++ wrapper
   - Executes via ModelNew with modified forward signature

### 3. TRITON (Triton JIT Kernels)
**Purpose**: Triton language kernels with automatic optimization

**Compilation Process**:
1. **Module Loading** (`_TritonModuleLoader`):
   - Creates temporary Python module
   - Executes source to populate namespace
   - Identifies `@triton.jit` decorated functions

2. **Kernel Discovery**:
   - Searches for `JITFunction` instances
   - Supports named kernel selection
   - Falls back to first kernel found

3. **Execution Configuration**:
   - Extracts grid from IOContract
   - Handles meta parameters (constexpr)
   - Configures num_warps/num_stages

**Requirements**:
- **MANDATORY**: IOContract with complete specifications
- Must have `@triton.jit` decorated kernel
- Meta parameters marked with `is_meta=true`

**IOContract Example**:
```python
io_contract = IOContract(
    args=[
        ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=[1024], dtype="float32")),
        ArgSpec(name="y", type="tensor", role="output", tensor_spec=TensorSpec(shape=[1024], dtype="float32")),
        ArgSpec(name="BLOCK_SIZE", type="int", value=128, is_meta=True)
    ],
    launch=LaunchConfig(
        grid=LaunchDim(x=8),
        num_warps=4,
        num_stages=2
    )
)
```

### 4. CUDA
**Purpose**: Direct CUDA C/C++ kernels

**Compilation**:
- Uses CuPy for compilation
- Requires IOContract for input/output specs
- Direct kernel launch without wrapper

### 5. MULTI_KERNEL (Python Scripts with Mixed Kernel Types)

**Purpose**: Execute Python scripts that combine multiple kernel types (CUDA, Triton, PyTorch) in a single workflow.

**Key Features**:
- Supports mixed kernel types in one script
- Flexible execution with explicit entry point function
- Full Python environment with all server-installed packages
- Ideal for multi-stage pipelines and hybrid optimizations

**Requirements**:
- **MANDATORY**: IOContract with complete input specifications
- **MANDATORY**: `entry_point` in metadata (function name to execute)
- Entry point function must exist in source code
- Uses server-installed packages (Triton, CuPy, PyTorch, etc.)

**Metadata Structure**:
```python
{
  "entry_point": "run",        # Required: function name to call
  "description": "..."         # Optional: description of the workflow
}
```

**Example 1: Simple Multi-Kernel (Triton + PyTorch)**

```python
# Source code combining Triton and PyTorch
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for element-wise addition"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def run(x, y):
    """Entry point: add tensors using Triton, then multiply by 2 using PyTorch"""
    output = torch.empty_like(x)
    n_elements = x.numel()

    # Launch Triton kernel for addition
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # PyTorch operation for multiplication
    output = output * 2

    return output
```

**Request Format**:
```json
{
  "kernel": {
    "source_code": "...",
    "kernel_type": "multi_kernel",
    "metadata": {
      "entry_point": "run",
      "description": "Add with Triton, multiply with PyTorch"
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
  },
  "num_trials": 10
}
```

**Example 2: Complex Multi-Kernel (Triton + CUDA + PyTorch)**

```python
import torch
import triton
import triton.language as tl
import cupy as cp

# Triton kernel for preprocessing
@triton.jit
def preprocess_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = x * 2.0 + 1.0
    tl.store(output_ptr + offsets, result, mask=mask)

# CUDA kernel for core computation
cuda_source = r"""
extern "C" __global__
void compute_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}
"""
compute_kernel = cp.RawKernel(cuda_source, "compute_kernel")

def run(x):
    """Entry point: preprocess with Triton, compute with CUDA, postprocess with PyTorch"""
    n_elements = x.numel()

    # Stage 1: Triton preprocessing
    preprocessed = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    preprocess_kernel[grid](x, preprocessed, n_elements, BLOCK_SIZE=256)

    # Stage 2: CUDA computation
    computed = torch.empty_like(preprocessed)
    block_size = 256
    grid_size = (n_elements + block_size - 1) // block_size
    compute_kernel((grid_size,), (block_size,),
                   (cp.asarray(preprocessed), cp.asarray(computed), n_elements))

    # Stage 3: PyTorch postprocessing
    result = computed + torch.mean(computed)

    return result
```

**Error Handling**:
- **Full stack traces** returned on failures for debugging
- **Runtime errors** include complete traceback
- **Module loading errors** clearly reported with context
- **Entry point not found** returns descriptive error

**Common Error Messages**:
- `"Entry point function 'run' not found in module"` - Function name mismatch
- `"IOContract is required for multi_kernel type"` - Missing input specifications
- `"entry_point must be specified in metadata"` - Missing metadata field

**Common Use Cases**:
1. **Multi-stage pipelines**: Preprocess → Compute → Postprocess
2. **Hybrid optimization**: Combine Triton and custom CUDA for specific bottlenecks
3. **Performance comparisons**: Test different kernel implementations in single script
4. **Complex workflows**: Integrate multiple specialized kernels efficiently

**Compilation Process**:
1. **Module Loading**: Source code loaded as temporary Python module
2. **Entry Point Extraction**: Function specified in metadata is extracted
3. **Input Generation**: Inputs created from IOContract specification
4. **Execution**: Entry point function called with generated inputs
5. **Result Collection**: Outputs captured and validated

**Limitations**:
- Entry point function must accept positional arguments matching IOContract order
- All kernel types must be supported by server environment
- No support for external file dependencies
- Module imports must be available in server Python environment

## Source Code Constraints & Guidelines

### General Source Code Rules

All kernel source code must follow these fundamental rules:

1. **Self-Contained**: Include ALL necessary imports at the top
2. **No Execution Code**: Remove `if __name__ == "__main__"` blocks
3. **Clean Namespace**: Avoid global variables that execute on import
4. **Minimal Code**: Only include kernel-related code, no test harnesses
5. **No Relative Imports**: Use absolute imports only
6. **String Format**: Source is executed via `exec()` - ensure proper escaping

### TORCH Kernel Constraints

**Purpose**: Pure PyTorch reference implementations without any CUDA/Triton code.

#### ✅ ACCEPTABLE TORCH Source

```python
# GOOD: Clean PyTorch model with proper imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 128)
    
    def forward(self, x):
        return F.relu(self.linear(x))

# Multiple functions are OK
def preprocess(x):
    return x * 2.0

def custom_activation(x):
    return torch.sigmoid(x) * x

# Targeting specific function via metadata
def my_kernel(x, y):
    return torch.matmul(x, y) + torch.relu(x)
```

#### ❌ NOT ACCEPTABLE TORCH Source

```python
# BAD: Mixed kernel types
import torch
import triton  # NO! Don't mix Triton with TORCH

class Model(nn.Module):
    def forward(self, x):
        # NO! CUDA code in TORCH kernel
        cuda_kernel = '''
        __global__ void kernel() {}
        '''
        return x

# BAD: Execution code
model = Model()  # NO! Don't instantiate
result = model(torch.randn(10))  # NO! Don't execute

# BAD: Test code
if __name__ == "__main__":  # NO! Remove test code
    test_model()
```

**Metadata for Function Targeting**:
```json
{
  "function_name": "my_kernel",  // Targets standalone function
  "class_name": "Model",  // Or target class
  "method_name": "forward"        // And specific method
}
```

### TORCH_CUDA Kernel Constraints

**Purpose**: PyTorch models with embedded CUDA kernels using `cpp_extension.load_inline`.

#### ✅ ACCEPTABLE TORCH_CUDA Source

```python
# GOOD: Proper TORCH_CUDA structure
# Include used torch imports  
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source as string variable
# - Include C++/CUDA headers 


cuda_source = """
#include <torch/extension.h> 
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

# C++ wrapper as string for Torch interface with
cpp_source = '''
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
'''

# EXACTLY ONE load_inline call which creates `my_namespace` module
my_namespace = load_inline(
    name='my_cuda_kernel',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['forward']
)

# Expects `ModelNew` class w/ forward function which uses `my_namespace` module  
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.my_namespace = my_namespace

    def forward(self, x):
        return self.my_namespace.forward(x)
```

#### ❌ NOT ACCEPTABLE TORCH_CUDA Source

```python
# BAD: Multiple load_inline calls
module1 = load_inline(...)  # NO! Only one allowed
module2 = load_inline(...)  # This will fail extraction

# BAD: Inline strings instead of variables
load_inline(
    cuda_sources='''kernel code''',  # NO! Use variables
    cpp_sources='''wrapper code'''
)

# BAD: Missing required parameters
load_inline(
    name='kernel',
    cuda_sources=cuda_code  # Missing cpp_sources!
)

# BAD: Wrong variable pattern
cuda_kernel_source = cuda_code  # NO! Extractor looks for specific names
cpp_wrapper_source = cpp_code   # Must match extraction pattern

# BAD: Execution or compilation code
module = load_inline(...)
module.forward(torch.randn(10))  # NO! Don't execute

# BAD: Mixed with Triton
import triton  # NO! Keep kernel types separate
```

**Critical Requirements**:
- MUST use variable names that match extraction pattern
- MUST have `cpp_sources` and `cuda_sources` parameters
- MUST have `functions` parameter with C++ function names
- Variables must be defined before `load_inline` call

### TRITON Kernel Constraints

**Purpose**: Triton JIT kernels with automatic optimization.

#### ✅ ACCEPTABLE TRITON Source

```python
# GOOD: Clean Triton kernel
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

# Multiple kernels OK - specify which via metadata
@triton.jit
def vector_mul_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Different kernel implementation
    pass

# Utility functions are OK
def calculate_grid(n, block_size):
    return (n + block_size - 1) // block_size
```

#### ❌ NOT ACCEPTABLE TRITON Source

```python
# BAD: Mixed with PyTorch models
import torch.nn as nn

class Model(nn.Module):  # NO! Don't mix models with Triton
    def forward(self, x):
        pass

@triton.jit
def kernel(...):
    pass

# BAD: Execution code
def launch_kernel(x, y):  # NO! Don't include launchers
    grid = (1024,)
    kernel[grid](x, y)
    
# BAD: Test code
if __name__ == "__main__":
    test_triton_kernel()  # NO! Remove test code

# BAD: Missing triton.jit decorator
def my_kernel(...):  # Must have @triton.jit
    pass

# BAD: No IOContract (Triton REQUIRES IOContract)
# Server cannot execute without IOContract specification
```

**IOContract Requirement**:
```json
{
  "io": {
    "args": [
      {"name": "x_ptr", "type": "tensor", "tensor_spec": {...}},
      {"name": "BLOCK_SIZE", "type": "int", "value": 128, "is_meta": true}
    ],
    "launch": {
      "grid": {"x": 8},
      "num_warps": 4
    }
  }
}
```

### CUDA Kernel Constraints

**Purpose**: Raw CUDA C/C++ kernels (in development).

#### ✅ ACCEPTABLE CUDA Source

```c
// GOOD: Pure CUDA C/C++
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__device__ float device_func(float x) {
    return x * x;
}

// Multiple kernels OK
__global__ void vector_mul(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
```

#### ❌ NOT ACCEPTABLE CUDA Source

```c
// BAD: Python code mixed in
import torch  // NO! This is C/C++, not Python

// BAD: Main function
int main() {  // NO! Don't include main
    return 0;
}

// BAD: Host code
void launch_kernel() {  // NO! Only kernel code
    vector_add<<<1, 1>>>();
}

// BAD: Missing __global__ qualifier
void my_kernel() {  // Must be __global__ or __device__
}
```

### Multiple Kernels in Single Source

The server supports multiple kernels/functions in a single source file, but **only one will be executed**.

#### Kernel Selection Mechanisms

**TORCH**: Use metadata to specify target
```python
# Source with multiple options
def kernel_a(x): return x * 2
def kernel_b(x): return x + 1
class Model:
    def method_a(self, x): return x
    def method_b(self, x): return x * x

# Metadata selects execution target
metadata = {
    "function_name": "kernel_a"  # Executes kernel_a
    # OR
    "class_name": "Model",
    "method_name": "method_b"     # Executes Model.method_b
}
```

**TRITON**: Named kernel selection (optional) or first kernel found
```python
@triton.jit
def kernel_1(...): pass

@triton.jit
def kernel_2(...): pass

# Option 1: Specify kernel name in metadata (optional)
metadata = {
    "kernel_name": "kernel_2"  # Explicitly select kernel_2
}

# Option 2: Omit metadata (uses automatic selection)
# Without metadata: Automatically executes kernel_1 (first @triton.jit decorated function found)
```

**TORCH_CUDA**: Only the extracted CUDA kernel is compiled
```python
# The load_inline defines what gets compiled
module = load_inline(
    functions=['forward']  # Only 'forward' is available
)
```

**CUDA**: Specify kernel name in launch configuration
```c
__global__ void kernel_a(...) {}
__global__ void kernel_b(...) {}

// IOContract specifies which kernel
// kernel_name in metadata or launch config
```

### IOContract Seeding for Reproducibility

Use seeded tensor initialization for deterministic validation across kernels.

#### Seeded Tensor Generation

```json
{
  "io": {
    "args": [
      {
        "name": "input",
        "type": "tensor",
        "tensor_spec": {
          "shape": [32, 64],
          "dtype": "float32",
          "init": {
            "kind": "randn",
            "seed": 42,      // Ensures reproducibility
            "mean": 0.0,
            "std": 1.0
          }
        }
      },
      {
        "name": "weights", 
        "type": "tensor",
        "tensor_spec": {
          "shape": [64, 128],
          "dtype": "float32",
          "init": {
            "kind": "uniform",
            "seed": 123,     // Different seed for different tensor
            "low": -0.5,
            "high": 0.5
          }
        }
      }
    ]
  }
}
```

#### Validation Consistency

The server ensures both reference and custom kernels receive **identical inputs**:

1. **Seed-based Generation**: Same seed → same tensor values
2. **Validation Trials**: Each trial uses different seed for robustness
3. **Cross-kernel Consistency**: Both kernels get exact same inputs

Example validation flow:
```python
# Trial 1: seed=42
ref_output = ref_kernel(inputs_seed_42)
custom_output = custom_kernel(inputs_seed_42)
compare(ref_output, custom_output)

# Trial 2: seed=84  
ref_output = ref_kernel(inputs_seed_84)
custom_output = custom_kernel(inputs_seed_84)
compare(ref_output, custom_output)
```

### Common Pitfalls When Compiling from Strings

#### 1. Execution Code in Global Scope
```python
# BAD: This executes immediately
model = Model()
x = torch.randn(10)
result = model(x)  # NO! This runs during compilation

# GOOD: Only definitions
class Model:
    def forward(self, x):
        return x
```

#### 2. Missing Essential Imports
```python
# BAD: Assumes imports
def forward(x):
    return F.relu(x)  # NameError: F not defined

# GOOD: Explicit imports
import torch.nn.functional as F
def forward(x):
    return F.relu(x)
```

#### 3. Namespace Pollution
```python
# BAD: Modifies global state
import sys
sys.path.append('/my/path')  # NO! Affects server

# BAD: Global variables with side effects
COUNTER = 0
def increment():
    global COUNTER
    COUNTER += 1  # NO! Stateful globals
```

#### 4. Mixing Incompatible Kernel Types
**KernelType**: TRITON

```python
# BAD: TORCH kernel with Triton code
import torch
import triton 

@triton.jit
def kernel(): pass

class Model(torch.nn.Module): pass  # Confusing!
```

#### 5. Incorrect String Escaping
```python
# BAD: Unescaped quotes in CUDA string
cuda_source = "printf("Hello");'  # Syntax error

# GOOD: Proper escaping
cuda_source = '''printf("Hello");'''
# OR
cuda_source = "printf(\"Hello\");"
```

#### 6. Side Effects During Compilation
```python
# BAD: I/O operations
with open('file.txt', 'w') as f:  # NO! No file I/O
    f.write('data')

# BAD: Network requests  
import requests
data = requests.get('http://...')  # NO! No network

# BAD: Print statements (clutter logs)
print("Debug info")  # Avoid prints
```

### Debugging Source Code Issues

#### Step 1: Validate Syntax Locally
```python
# Test your source compiles
source = "your kernel source here"
try:
    compile(source, '<string>', 'exec')
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
```

#### Step 2: Check Imports
```python
# Verify all imports work
namespace = {}
exec(source, namespace)
# Check expected objects exist
assert 'Model' in namespace  # For TORCH
assert 'load_inline' in namespace    # For TORCH_CUDA
```

#### Step 3: Test Compilation Separately
```bash
# Test just compilation without validation/profiling
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "kernel": {
      "source_code": "...",
      "kernel_type": "torch",
      "io": {...}
    },
    "num_trials": 1
  }'
```

#### Common Error Messages

**"No load_inline call found"** (TORCH_CUDA)
- Check: Is `load_inline` present?
- Check: Is it correctly formatted?
- Check: Are cuda_sources/cpp_sources variables defined?

**"No @triton.jit kernels found"** (TRITON)
- Check: Is @triton.jit decorator present?
- Check: Are triton imports included?

**"Class 'Model' not found"** (TORCH)
- Check: Is class defined in source?
- Check: Is class name spelled correctly?
- Check: No syntax errors preventing execution?

**"IOContract is required for Triton"**
- Triton kernels MUST have IOContract
- Check: Is "io" field present in request?
- Check: Are all arguments specified?

**"Failed to extract CUDA kernel"** (TORCH_CUDA)
- Check: Variable names match pattern
- Check: load_inline has all required params
- Check: Only one load_inline call

## Compilation Process

### Backend Pattern Architecture
```python
backends = {
    KernelType.TORCH: TorchCompilationBackend(),
    KernelType.TORCH_CUDA: TorchCudaCompilationBackend(),
    KernelType.TRITON: TritonCompilationBackend(),
    KernelType.CUDA: CudaCompilationBackend(),
    KernelType.MULTI_KERNEL: MultiKernelCompilationBackend()
}
```

### Compilation Service Flow
1. **Request Reception**: `CompilationRequest` with `KernelCode`
2. **Backend Selection**: Based on `kernel_type`
3. **Device Setting**: `torch.cuda.set_device(gpu_id)`
4. **Kernel Compilation**: Backend-specific process
5. **Executable Creation**: Returns `BaseExecutableKernel`

### Error Handling

**Graceful Failure Philosophy**:

The server implements a critical design pattern: **kernel failures are not server errors**.

**HTTP Status Code Strategy**:

- **200 (Success)** - Request successfully processed, check response fields for kernel status:
  - `compiled=false` + `compilation_error="..."` - Kernel compilation failed
  - `correctness=false` + `validation_error="..."` - Kernel validation failed
  - Both successful compilation and validation indicated by `compiled=true` and `correctness=true`

- **400 (Client Error)** - Invalid request format:
  - Missing required fields (e.g., `source_code`, `kernel_type`)
  - Invalid IOContract structure
  - Malformed metadata
  - Type validation errors

- **500 (Server Error)** - Server infrastructure failure:
  - Job timeout (exceeded configured timeout)
  - Subprocess crash without state recovery
  - GPU resource allocation failure
  - Internal server errors (database, file system, etc.)

**Why This Design Matters**:

This pattern allows clients to programmatically distinguish between three categories of issues:

1. **User's kernel has bugs** (HTTP 200 with error flags)
   - Action: Fix kernel source code
   - Example: CUDA syntax error, missing function, incorrect logic

2. **User's request is malformed** (HTTP 400)
   - Action: Fix request structure or IOContract
   - Example: Missing required field, invalid JSON

3. **Server has problems** (HTTP 500)
   - Action: Retry request or contact administrator
   - Example: GPU unavailable, timeout, infrastructure failure

**Implementation Details**:
- Compilation failures return `CompilationResult(compiles=False, compilation_error="...")`
- Validation failures return `ValidationResult(correct=False, error="...")`
- Job continues to build response with all available information, even on partial failure
- Partial results preserved and returned when possible

### TORCH_CUDA Specific Details

**Kernel Extraction**:
```python
# Searches for pattern:
cpp_extension.load_inline(
    name="kernel_name",
    cpp_source=cpp_code,
    cuda_source=cuda_code,
    functions=["func1", "func2"]
)
```

**CuPy Compilation**:
```python
# Preprocesses CUDA source
processed_cuda = self._preprocess_cuda_source(cuda_source)
# Compiles with CuPy
kernel = cupy.RawKernel(processed_cuda, kernel_name)
```

**C++ Transformation**:
- Original: `void forward(torch::Tensor x, torch::Tensor y)`
- Transformed: `void forward(void* func_ptr, torch::Tensor x, torch::Tensor y)`
- Injects: `((kernel_fn)func_ptr)(grid, block, args, shared_mem, stream)`

## Validation Mechanisms

### Correctness Validation (Two Kernels)
**Purpose**: Verify custom kernel produces same output as reference

**Process**:
1. **Input Generation**:
   - Priority: `kernel._default_inputs` > IOContract > Error
   - Ensures identical inputs for both kernels
   - Device synchronization

2. **Warmup Phase**:
   - Single execution to trigger JIT compilation
   - Prevents timing artifacts

3. **Trial Execution**:
   - Deterministic seeding per trial
   - NVTX range markers for profiling
   - Output comparison with tolerance

4. **Tolerance Settings**:
   ```python
   torch.allclose(ref_output, custom_output, atol=1e-2, rtol=1e-2)
   ```

**NVTX Markers**:
- Reference: `{job_id}_original`
- Custom: `{job_id}_custom`
- Used by NCU for device metrics

### Executable Validation (Single Kernel)
**Purpose**: Verify kernel can execute without errors

**Process**:
1. Input generation from IOContract
2. Multiple trial executions
3. Pass if all trials complete without exception

### Input Selection Logic
```python
def _select_inputs(kernel, device, default=None):
    # 1. Use kernel's default inputs if available
    if kernel._default_inputs:
        return kernel._default_inputs
    
    # 2. Generate from IOContract
    if kernel.io_contract:
        manager = IOContractManager()
        inputs = manager.generate_inputs(kernel.io_contract, device)
        # Filter meta params for Triton
        if kernel.kernel_type == "triton":
            inputs = [inp for i, inp in enumerate(inputs) 
                     if not kernel.io_contract.args[i].is_meta]
        return inputs
    
    # 3. Use provided default or error
    if default:
        return default
    raise RuntimeError("No inputs available")
```

## Profiling Implementation

### Profiling Methods

#### 1. CUDA Graphs (Preferred)
**Advantages**:
- Eliminates CPU overhead
- Consistent timing measurements
- Optimal for repeated execution

**Implementation**:
```python
# Graph capture on non-default stream
capture_stream = torch.cuda.Stream(device)
with torch.cuda.stream(capture_stream):
    # Warmup
    for _ in range(3):
        kernel(*static_args)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=capture_stream):
    output = kernel(*static_args)

# Replay with timing
for trial in range(num_trials):
    start_event.record()
    g.replay()
    end_event.record()
```

**Critical Details**:
- Uses **non-default stream** for capture
- Static input buffers required
- Falls back to events on failure

#### 2. CUDA Events (Fallback)
**Used When**:
- Graph capture fails
- Dynamic shapes
- Complex control flow

**Implementation**:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
kernel(*inputs)
end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)
```

### Runtime Statistics Calculation
```python
stats = {
    "mean": np.mean(times),
    "std": np.std(times),
    "min": np.min(times),
    "max": np.max(times),
    "median": np.median(times),
    "percentile_95": np.percentile(times, 95),
    "percentile_99": np.percentile(times, 99)
}
```

### Device Metrics Collection

**NCU Integration**:
- Wrapper command: `ncu --export report.ncu-rep --nvtx --nvtx-include {job_id}*`
- Sections: SpeedOfLight, Occupancy, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis
- Parser extracts metrics into structured format

**Metrics Categories**:
1. **Speed of Light**: Throughput percentages
2. **Occupancy**: Warps, blocks, registers
3. **Memory**: DRAM bandwidth, cache hit rates
4. **Compute**: Pipeline utilization
5. **Detailed**: IPC, active cycles

## IOContract Specification

### Core Components

#### TensorSpec
```python
@dataclass
class TensorSpec:
    shape: List[int]           # Required for generation
    dtype: str = "float32"      # PyTorch dtype string
    init: Optional[TensorInit]  # Server-side generation
    data: Optional[TensorData]  # Client-provided data
```

#### TensorInit Options
- `"randn"`: Normal distribution (mean, std)
- `"zeros"`: Zero tensor
- `"ones"`: One tensor
- `"uniform"`: Uniform distribution (low, high)
- `"full"`: Constant value (fill_value)
- `"arange"`: Sequential values (start, step)

#### ArgSpec
```python
@dataclass
class ArgSpec:
    name: str                   # Argument name
    type: str                   # "tensor", "int", "float", "bool", "str"
    value: Optional[Any]        # For scalars
    tensor_spec: Optional[TensorSpec]  # For tensors
    role: str = "input"         # "input", "output", "inout"
    is_meta: bool = False       # Triton constexpr parameter
```

#### LaunchConfig
```python
@dataclass
class LaunchConfig:
    grid: Optional[LaunchDim]   # Grid dimensions (x, y, z)
    block: Optional[LaunchDim]  # Block dimensions (CUDA only)
    num_warps: Optional[int]    # Triton only
    num_stages: Optional[int]   # Triton only
```

### Triton-Specific Requirements

**Meta Parameters**:
- Mark with `is_meta=True`
- Provide constant value
- Not passed as regular arguments
- Used for kernel specialization

**Example**:
```python
ArgSpec(name="BLOCK_SIZE", type="int", value=128, is_meta=True)
```

### Tensor Data Transfer

**Compression Support**:
```python
TensorData(
    data_b64="...",  # Base64 encoded bytes
    compress="zlib",  # Compression method
    shape=[32, 64],
    dtype="float32"
)
```

### IOContract Helper Utilities

The server provides helper functions and builders to simplify IOContract creation, located in `io_contract/spec_builders.py`.

#### Tensor Spec Helper Functions

These helper functions create `TensorSpec` objects with various initialization methods:

**create_randn_spec()** - Random normal distribution
```python
from io_contract.spec_builders import create_randn_spec

tensor_spec = create_randn_spec(
    shape=[1024, 512],
    dtype="float32",
    seed=42,         # Optional: for reproducibility
    mean=0.0,        # Optional: default 0.0
    std=1.0          # Optional: default 1.0
)
```

**create_uniform_spec()** - Uniform distribution
```python
from io_contract.spec_builders import create_uniform_spec

tensor_spec = create_uniform_spec(
    shape=[1024],
    dtype="float32",
    seed=42,         # Optional: for reproducibility
    low=0.0,         # Optional: default 0.0
    high=1.0         # Optional: default 1.0
)
```

**create_zeros_spec()** - All zeros
```python
from io_contract.spec_builders import create_zeros_spec

tensor_spec = create_zeros_spec(shape=[1024, 512], dtype="float32")
```

**create_ones_spec()** - All ones
```python
from io_contract.spec_builders import create_ones_spec

tensor_spec = create_ones_spec(shape=[1024, 512], dtype="float32")
```

**create_full_spec()** - Constant value
```python
from io_contract.spec_builders import create_full_spec

tensor_spec = create_full_spec(
    shape=[1024],
    fill_value=3.14,
    dtype="float32"
)
```

**create_arange_spec()** - Sequential values
```python
from io_contract.spec_builders import create_arange_spec

tensor_spec = create_arange_spec(
    shape=[1024],
    dtype="float32",
    start=0.0,       # Optional: default 0.0
    step=1.0         # Optional: default 1.0
)
```

#### IOContractBuilder - Fluent API

Build complete IOContracts using method chaining for cleaner, more readable code:

**Basic Example**:
```python
from io_contract.spec_builders import IOContractBuilder, create_randn_spec

contract = (
    IOContractBuilder()
    .add_input_tensor("x", create_randn_spec([1024], seed=42))
    .add_output_tensor("y", shape=[1024], dtype="float32")
    .build()
)
```

**Complex Example with Triton Meta Parameters**:
```python
from io_contract.spec_builders import IOContractBuilder, create_randn_spec, create_uniform_spec
from shared.models import LaunchConfig, LaunchDim

contract = (
    IOContractBuilder()
    # Input tensors
    .add_input_tensor("x", create_randn_spec([1024, 512], dtype="float32", seed=42))
    .add_input_tensor("weights", create_uniform_spec([512, 256], seed=43, low=-0.5, high=0.5))

    # Output tensor
    .add_output_tensor("result", shape=[1024, 256], dtype="float32")

    # Scalar parameters
    .add_scalar("alpha", "float", 2.0)
    .add_scalar("n_elements", "int", 1024)

    # Meta parameters (Triton constexpr)
    .add_meta_param("BLOCK_SIZE", value=256)
    .add_meta_param("NUM_STAGES", value=2)

    # Launch configuration
    .set_launch_config(
        LaunchConfig(
            grid=LaunchDim(x=4, y=1, z=1),
            num_warps=4,
            num_stages=2
        )
    )

    .build()
)
```

**Available Builder Methods**:

| Method | Description | Parameters |
|--------|-------------|------------|
| `add_input_tensor(name, tensor_spec)` | Add input tensor | `name: str`, `tensor_spec: TensorSpec` |
| `add_output_tensor(name, shape, dtype)` | Add output tensor | `name: str`, `shape: List[int]`, `dtype: str = "float32"` |
| `add_inout_tensor(name, tensor_spec)` | Add input/output tensor | `name: str`, `tensor_spec: TensorSpec` |
| `add_scalar(name, type, value)` | Add scalar parameter | `name: str`, `type: str`, `value: Union[int, float, str, bool]` |
| `add_meta_param(name, value)` | Add meta parameter (Triton) | `name: str`, `value: int` |
| `set_launch_config(launch_config)` | Set launch configuration | `launch_config: LaunchConfig` |
| `build()` | Create IOContract | Returns: `IOContract` |

#### JSON Serialization Helpers

These utilities help with saving, loading, and serializing IOContracts:

**to_json()** - Serialize to JSON string
```python
from io_contract.spec_builders import to_json

json_str = to_json(contract, pretty=True)
print(json_str)
```

**save_to_file()** - Save to JSON file
```python
from io_contract.spec_builders import save_to_file

save_to_file(contract, "my_contract.json", pretty=True)
```

**load_from_file()** - Load from JSON file
```python
from io_contract.spec_builders import load_from_file

contract = load_from_file("my_contract.json", obj_type="IOContract")
```

#### Complete Workflow Example

Putting it all together - creating a complete kernel request with IOContract helpers:

```python
from io_contract.spec_builders import (
    IOContractBuilder,
    create_randn_spec,
    create_zeros_spec,
    to_json
)
from shared.models import LaunchConfig, LaunchDim, KernelCode

# Build IOContract using helpers
io_contract = (
    IOContractBuilder()
    .add_input_tensor("x_ptr", create_randn_spec([4096], dtype="float32", seed=42))
    .add_input_tensor("y_ptr", create_randn_spec([4096], dtype="float32", seed=43))
    .add_output_tensor("z_ptr", shape=[4096], dtype="float32")
    .add_scalar("n_elements", "int", 4096)
    .add_meta_param("BLOCK_SIZE", value=256)
    .set_launch_config(
        LaunchConfig(
            grid=LaunchDim(x=16),  # 4096 / 256 = 16 blocks
            num_warps=4
        )
    )
    .build()
)

# Create kernel code
kernel_code = KernelCode(
    source_code=triton_kernel_source,
    kernel_type="triton",
    io=io_contract
)

# Serialize for API request
request_json = to_json({
    "kernel": kernel_code.dict(),
    "num_trials": 100
}, pretty=True)
```

**Supported Initialization Methods**:

| Method | Description | Parameters |
|--------|-------------|------------|
| `randn` | Random normal distribution | `seed`, `mean`, `std` |
| `uniform` | Uniform distribution [low, high) | `seed`, `low`, `high` |
| `zeros` | All zeros | None |
| `ones` | All ones | None |
| `full` | Fill with constant value | `fill_value` |
| `arange` | Sequential values (0, 1, 2, ...) | `start`, `step` |

## API Endpoints

### POST /compare
**Purpose**: Compare two kernels (reference vs custom)

**Request Format**:
```json
{
  "ref_kernel": {
    "source_code": "...",
    "kernel_type": "torch",
    "io": { /* Optional IOContract */ }
  },
  "custom_kernel": {
    "source_code": "...",
    "kernel_type": "torch_cuda",
    "io": { /* Optional IOContract */ }
  },
  "num_trials": 100,
  "timeout": 120
}
```

**Response Format**:
```json
{
  "job_id": "uuid",
  "kernel_exec_result": {
    "compiled": true,
    "correctness": true,
    "runtime": 1.234,  // Mean runtime in ms
    "runtime_stats": { /* Detailed stats */ },
    "metadata": {
      "gpu_id": 0,
      "device_metrics": { /* Optional NCU metrics */ }
    }
  },
  "ref_runtime": { /* Reference kernel stats */ },
  "status": "success"
}
```

### POST /evaluate
**Purpose**: Evaluate single kernel performance

**Request Format**:
```json
{
  "kernel": {
    "source_code": "...",
    "kernel_type": "triton",
    "io": { /* Required for Triton */ }
  },
  "num_trials": 100,
  "timeout": 120
}
```

### GET /health
Returns server health status and available backends

### GET /stats
Returns comprehensive server statistics:
- Job statistics
- Compilation metrics
- Profiling metrics
- GPU utilization
- Throughput metrics

## File Structure

```
cuda_eval_server_v2/
├── app.py                          # FastAPI application
├── main.py                         # Server entry point
├── mcp_server.py                   # MCP server implementation
├── subprocess_worker.py            # Isolated worker process
│
├── orchestration/
│   └── job_manager.py              # Job orchestration, GPU management
│
├── compilation/
│   ├── compiler_service.py         # Backend dispatcher
│   ├── base_compiler.py            # Base backend interface
│   ├── torch/
│   │   ├── __init__.py
│   │   └── torch_backend.py        # Pure PyTorch backend
│   ├── torch_cuda/
│   │   ├── __init__.py
│   │   ├── torch_cuda_backend.py   # PyTorch + CUDA backend
│   │   ├── compiler.py
│   │   ├── kernel_extractor.py
│   │   └── cpp_wrapper_transformer.py
│   ├── triton/
│   │   ├── __init__.py
│   │   └── triton_backend.py       # Triton backend (includes TritonExecutableKernel)
│   ├── cuda/
│   │   ├── __init__.py
│   │   └── cuda_backend.py         # Raw CUDA backend
│   └── multi_kernel/
│       ├── __init__.py
│       └── multi_kernel_backend.py # Multi-kernel backend
│
├── validation/
│   ├── base_validator.py           # Abstract validator base
│   └── correctness_validator.py    # Validators (Correctness, Executable)
│
├── profiling/
│   └── kernel_profiler.py          # CUDA graphs, statistical analysis
│
├── shared/
│   ├── models.py                   # Pydantic models (KernelCode, Response, etc.)
│   ├── executable_kernels.py       # Executable kernel wrappers
│   ├── kernel_metadata.py          # Kernel metadata classes
│   ├── metrics_collector.py        # Metrics collection system
│   ├── gpu_resource_manager.py     # GPU allocation
│   ├── device_metrics_parser.py    # NCU report parsing
│   └── utils.py                    # Utilities
│
├── io_contract/
│   ├── __init__.py                 # Module exports
│   ├── manager.py                  # IOContract management
│   ├── tensor_utils.py             # Tensor generation utilities
│   └── spec_builders.py            # Helper builders for IOContract specs ★
│
├── client/                         # Python client library
│   ├── __init__.py
│   ├── setup.py                    # Client package setup
│   ├── kernel_eval_client/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── models.py
│   │   ├── specs.py
│   │   └── builder.py
│   └── examples/
│       ├── simple_pytorch.py
│       ├── triton_kernel.py
│       ├── cuda_kernel.py
│       └── advanced_patterns.py
│
├── tests/
│   ├── conftest.py                 # pytest configuration
│   ├── fixtures/
│   │   ├── __init__.py
│   │   ├── factories.py            # Test data factories
│   │   ├── kernels.py              # Kernel test fixtures
│   │   ├── validators.py           # Response validators
│   │   └── test_data_loader.py
│   ├── unit/                       # Fast, isolated component tests
│   │   ├── test_backends.py
│   │   ├── test_io_contract.py
│   │   ├── test_models.py
│   │   └── test_validators.py
│   ├── integration/                # Component interaction tests
│   │   ├── test_endpoints.py
│   │   ├── test_kernels.py
│   │   ├── test_failure.py
│   │   ├── test_multi_kernel.py
│   │   └── test_triton_metadata.py
│   ├── e2e/                        # End-to-end workflow tests
│   │   └── test_workflows.py
│   └── mcp/                        # MCP-specific tests
│       ├── fixtures/
│       ├── test_mcp_tools.py
│       ├── test_mcp_integration.py
│       └── test_mcp_error_handling.py
│
├── old_tests/                      # DEPRECATED: Legacy test scripts
│
├── scripts/
│   ├── docker-build-mcp.sh         # Build MCP Docker image
│   ├── docker-run-mcp.sh           # Run MCP Docker container
│   ├── start-mcp.sh                # Start MCP server
│   └── start-hybrid.sh             # Start hybrid mode server
│
├── test_data/                      # Test kernel source files
│
├── docker-build.sh                 # Build FastAPI Docker image
├── docker-run.sh                   # Run FastAPI Docker container
├── deploy.sh                       # Deployment script
├── Dockerfile                      # Base Docker image
├── Dockerfile.optimized            # Optimized Docker image
├── Dockerfile.mcp                  # MCP Docker image
├── docker-compose.yml              # Docker compose for FastAPI
├── docker-compose.mcp.yml          # Docker compose for MCP
│
└── Documentation:
    ├── README.md                   # Quick start and API overview
    ├── USER_MANUAL.md              # Detailed system design (this document)
    ├── API_GUIDE.md                # Complete API reference
    ├── TEST_GUIDE.md               # Comprehensive testing guide
    ├── DEVICE_METRICS_GUIDE.md     # NCU profiling usage
    ├── DEPLOYMENT_CLUSTER.md       # Kubernetes deployment
    ├── MCP_README.md               # MCP integration guide
    ├── MCP_QUICK_START.md          # MCP quick start
    └── CLAUDE.md                   # Claude Code AI assistant instructions
```

**Key Directory Highlights**:

- **compilation/multi_kernel/**: New backend for mixed kernel types
- **io_contract/spec_builders.py**: Helper utilities for building IOContracts
- **tests/**: Organized into unit/, integration/, e2e/, and mcp/ test suites
- **client/**: Python client library for programmatic API access
- **mcp_server.py**: Model Context Protocol integration for AI agents
- **old_tests/**: Deprecated legacy tests (use tests/ instead)

## Implementation Details

### Subprocess Worker Execution

**State Persistence**:
```python
def save_state(job_state, job_id):
    # Saves to /tmp/job_{job_id}_result.json
    # Preserves state across crashes
    # Parent process reads on completion
```

**Execution Flow**:
1. Load input from temp file
2. Set CUDA device
3. Execute pipeline (compile → validate → profile)
4. Save state after each phase
5. Return final state

### Error Recovery

The server implements comprehensive error recovery to maximize information returned to clients, even in failure scenarios.

**Partial Results** (see "Error Handling" section under "Compilation Process" for full philosophy):

The server follows the **graceful failure pattern** where kernel failures return HTTP 200 with error flags:

- **Subprocess crash with partial results** → Returns available data with appropriate error flags
  - State persistence allows recovery of intermediate results
  - Compilation results preserved even if validation crashes

- **Compilation failure** → HTTP 200 with `compiled=false` and `compilation_error` message
  - Full error traceback included for debugging
  - Reference kernel results included if available (for /compare endpoint)

- **Validation failure** → HTTP 200 with `compiled=true` but `correctness=false`
  - Kernel compiled successfully but produces incorrect results
  - Error details in `validation_error` field

- **Always attempts to return maximum information**:
  - Profiling results included if validation passed
  - Device metrics included if collection was enabled
  - Partial metadata always returned

**Subprocess Exit Codes**:
- **0**: Success (all phases completed)
- **1**: Initialization failure (setup error before execution)
- **-9**: Killed by OOM (out of memory)
- **-15**: Terminated (SIGTERM received)

**State Persistence**:

Subprocess workers save state to `/tmp/job_{job_id}_result.json` after each phase:
1. After compilation → Save compilation result
2. After validation → Save validation result
3. After profiling → Save profiling result

This allows parent process to recover maximum information even if subprocess crashes.

### Memory Management

**Cache Clearing**:
```python
torch.cuda.empty_cache()  # After validation/profiling
```

**Job Cleanup**:
- Background task every 5 minutes
- Removes jobs older than 1 hour
- Manual trigger via `/admin/cleanup-jobs`

### Metrics Collection

**Compilation Metrics**:
- Success rate
- Average compilation time
- Cache hit rate

**Profiling Metrics**:
- Average profiling time
- Graph capture success rate
- Correctness validation rate

**Throughput Metrics**:
- Requests per second
- Average request latency
- Active concurrent jobs

## Best Practices

### Kernel Source Code Guidelines

#### TORCH Kernels
- Use standard PyTorch operations
- Avoid custom CUDA code
- Ensure reproducible results

#### TORCH_CUDA Kernels
- Follow `cpp_extension.load_inline()` pattern exactly
- Keep CUDA kernel self-contained
- Ensure C++ wrapper matches kernel signature
- Test compilation separately first

#### TRITON Kernels
- **Always provide complete IOContract**
- Mark all constexpr parameters as meta
- Specify grid dimensions appropriately
- Use power-of-2 block sizes

### IOContract Best Practices

1. **Complete Specifications**:
   - Provide all tensor shapes
   - Specify correct dtypes
   - Mark output arguments properly

2. **Triton Requirements**:
   - All meta parameters must have values
   - Grid configuration must match kernel needs
   - num_warps typically 2, 4, or 8

3. **Performance Optimization**:
   - Use server-side generation for large tensors
   - Compress transferred data when possible
   - Reuse IOContract across requests

### Debugging Techniques

#### Enable Verbose Logging
```python
logging.basicConfig(level=logging.DEBUG)
```

#### Check Compilation Separately
Test compilation before full evaluation:
```python
# Just compile, don't validate/profile
response = requests.post("/evaluate", json={
    "kernel": kernel_code,
    "num_trials": 1  # Minimal profiling
})
```

#### NVTX Profiling
Use NCU with NVTX ranges:
```bash
ncu --nvtx --nvtx-include "job_id*" python your_script.py
```

#### Common Issues

**TORCH_CUDA Compilation Failures**:
- Check CUDA syntax errors
- Verify kernel function names match
- Ensure grid/block dimensions are valid
- Check pointer arithmetic

**TRITON IOContract Errors**:
- Verify all args have specifications
- Check meta parameter values
- Ensure tensor shapes match kernel expectations
- Verify grid dimensions

**Validation Failures**:
- Check numerical precision requirements
- Verify output tensor allocation
- Ensure deterministic operations
- Check for race conditions

### Performance Considerations

#### CUDA Graph Capture
- Best for static shapes
- Avoid dynamic control flow
- Warmup prevents JIT overhead

#### Profiling Accuracy
- Use sufficient trials (≥100)
- Consider variance in results
- Check for thermal throttling
- Verify GPU boost behavior

#### Device Metrics
- Enable with `ENABLE_DEVICE_METRICS=true`
- Specify sections with `NCU_SECTIONS`
- Significant overhead (~10x slower)
- Use sparingly for detailed analysis

## Advanced Features

### Custom Metadata
```python
kernel_metadata = {
    "algorithm": "gemm",
    "optimization": "tensor_cores",
    "precision": "tf32"
}
```

### Batch Evaluation
Process multiple kernels efficiently:
```python
# Submit multiple jobs
job_ids = []
for kernel in kernels:
    response = requests.post("/evaluate", json={"kernel": kernel})
    job_ids.append(response.json()["job_id"])

# Poll for completion
for job_id in job_ids:
    result = requests.get(f"/job/{job_id}")
```

### GPU Type Detection
Server automatically detects:
- A100: Ampere architecture
- H100: Hopper architecture
- H200: Hopper+ architecture

Returns in response header: `X-GPU-Type`

## Troubleshooting

### Server Won't Start
- Check CUDA installation: `nvidia-smi`
- Verify Python environment
- Check port 8000 availability

### Compilation Errors
- Review CUDA syntax
- Check compute capability
- Verify kernel signatures
- Inspect compilation logs

### Validation Failures
- Reduce tolerance if needed
- Check for non-deterministic ops
- Verify input generation
- Compare intermediate results

### Profiling Issues
- Disable CUDA graphs if unstable
- Increase warmup iterations
- Check for memory leaks
- Monitor GPU temperature

## Summary

The CUDA Evaluation Server V2 provides a robust, production-ready system for kernel compilation, validation, and profiling. Key strengths:

1. **Safety**: Subprocess isolation prevents crashes
2. **Flexibility**: Multiple kernel type support
3. **Accuracy**: CUDA graph profiling, NCU metrics
4. **Usability**: Comprehensive IOContract system
5. **Reliability**: Graceful failure handling

The server handles complex compilation patterns (TORCH_CUDA extraction/transformation), modern kernel languages (Triton), multi-kernel workflows, and provides detailed performance metrics. The IOContract system with helper utilities enables precise input/output specification while maintaining backward compatibility.

For production deployment, ensure proper GPU drivers, sufficient memory for compilation, and appropriate timeout settings based on kernel complexity.

## Related Documentation

This manual provides comprehensive technical details about the CUDA Evaluation Server V2 system architecture, kernel types, and implementation. For other aspects of the system, refer to these documents:

### Core Documentation
- **README.md** - Quick start guide, installation instructions, and basic API overview
- **API_GUIDE.md** - Complete API reference with detailed request/response examples and error codes
- **CLAUDE.md** - Instructions for Claude Code AI assistant when working with this codebase

### Operational Guides
- **TEST_GUIDE.md** - Testing infrastructure, how to run tests, and testing best practices
- **DEVICE_METRICS_GUIDE.md** - NCU profiling setup, device metrics collection, and interpretation
- **DEPLOYMENT_CLUSTER.md** - Kubernetes deployment, cluster setup, and scaling configuration

### Integration & Extensions
- **MCP_README.md** - Model Context Protocol integration guide for AI agents
- **MCP_QUICK_START.md** - Quick start tutorial for MCP integration
- **client/README.md** - Python client library documentation and usage examples

### Additional Resources
- **examples/** - Example kernel implementations and usage patterns
- **tests/** - Test suite with examples of valid kernel structures and IOContract usage
- **test_data/** - Sample kernel source files for reference

For the latest updates and issue reporting, visit the project repository.