# CUDA Evaluation Server V2 Architecture

## Overview

The CUDA Evaluation Server V2 implements a **backend pattern** architecture for flexible kernel compilation and evaluation. This design allows the server to support multiple kernel types (PyTorch, CUDA, Triton, etc.) through pluggable compilation backends.

## Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Backend Pattern**: Compilation logic is abstracted into backends for different kernel types
3. **Subprocess Isolation**: Dangerous operations run in isolated processes
4. **Async Processing**: Non-blocking request handling for better scalability
5. **Extensibility**: Easy to add new kernel types and compilation methods

## Component Architecture

### Request Flow

```
1. Client Request → FastAPI (app.py)
2. FastAPI → JobManager (orchestration/job_manager.py)
3. JobManager → Subprocess Worker (subprocess_worker.py)
4. Subprocess Worker orchestrates:
   - CompilationService → Backend Selection
   - ValidationService → Correctness Check
   - ProfilingService → Performance Measurement
5. Results → JobManager → FastAPI → Client Response
```

### Core Components

#### 1. FastAPI Frontend (`app.py`)
- **Responsibility**: HTTP request handling, API compatibility
- **Key Features**:
  - Async request processing
  - Backward compatibility with V1 API
  - Request validation using Pydantic models

#### 2. Job Manager (`orchestration/job_manager.py`)
- **Responsibility**: Workflow orchestration, job lifecycle management
- **Key Features**:
  - GPU resource acquisition in main process
  - Subprocess worker management
  - Log streaming from subprocess
  - Job state tracking

#### 3. Subprocess Worker (`subprocess_worker.py`)
- **Responsibility**: Isolated execution environment
- **Key Features**:
  - Runs compilation, validation, and profiling
  - Catches segfaults and crashes
  - Communicates via JSON files

#### 4. Compilation Service (`compilation/compiler_service.py`)
- **Responsibility**: Strategy selection and delegation
- **Key Features**:
  - Strategy registry for kernel types
  - Unified compilation interface
  - Error handling and metrics

#### 5. Compilation Strategies
- **Base Strategy** (`compilation/base_compiler.py`): Abstract interface
- **TorchCudaStrategy** (`compilation/torch_cuda/`): PyTorch + embedded CUDA
- **TorchStrategy** (`compilation/torch/`): Pure PyTorch models

#### 6. Validation Service (`validation/`)
- **Responsibility**: Kernel correctness validation
- **Key Features**:
  - Subprocess isolation for safety
  - Comparison with reference implementation
  - Statistical error analysis

#### 7. Profiling Service (`profiling/kernel_profiler.py`)
- **Responsibility**: Performance measurement
- **Key Features**:
  - CUDA graphs support
  - CUDA events fallback
  - Multiple trial statistics

## Data Models

### Kernel Type System

```python
class KernelType(Enum):
    TORCH = "torch"           # Pure PyTorch
    TORCH_CUDA = "torch_cuda" # PyTorch with embedded CUDA
    CUDA = "cuda"            # Raw CUDA (future)
    TRITON = "triton"        # Triton kernels (future)
```

### KernelCode Model

```python
@dataclass
class KernelCode:
    source_code: str
    kernel_type: KernelType
    metadata: Optional[Dict[str, Any]] = None
```

### BaseExecutableKernel

Abstract base class for executable kernels:
- Provides unified execution interface
- Handles input generation
- Supports profiling modes
- Device-aware execution

## Strategy Pattern Implementation

### Why Strategy Pattern?

1. **Open/Closed Principle**: Open for extension, closed for modification
2. **Single Responsibility**: Each strategy handles one kernel type
3. **Runtime Selection**: Choose strategy based on kernel type
4. **Easy Testing**: Test strategies independently

### Adding a New Strategy

1. Create strategy class:
```python
class MyStrategy(BaseCompilationStrategy):
    def compile(self, kernel_code: KernelCode, gpu_id: int):
        # Implementation
        pass
```

2. Register in CompilationService:
```python
self.strategies[KernelType.MY_TYPE] = MyStrategy()
```

## Subprocess Isolation

### Why Subprocess?

1. **Safety**: Kernel crashes don't affect server
2. **Resource Isolation**: Clean GPU state per evaluation
3. **Timeout Handling**: Kill hanging kernels
4. **Memory Management**: Prevent memory leaks

### Communication Protocol

1. Main process writes request to JSON file
2. Subprocess reads request, processes it
3. Subprocess writes result to JSON file
4. Main process reads result
5. Cleanup temporary files

## GPU Resource Management

### Centralized GPU Manager
- Single instance in main process
- Queue-based GPU allocation
- Timeout handling
- Resource cleanup on error

### GPU Lifecycle
1. JobManager acquires GPU
2. Passes GPU ID to subprocess
3. Subprocess uses specific GPU
4. JobManager releases GPU after completion

## Metrics Collection

### Types of Metrics
1. **Compilation Metrics**: Time, success rate, cache hits
2. **Profiling Metrics**: Runtime, speedup, GPU utilization
3. **System Metrics**: Request throughput, GPU usage
4. **Device Metrics**: NCU profiling (optional)

### NCU Integration
- Wraps subprocess with NCU command
- Collects Speed of Light metrics
- Parses NCU reports
- Includes in response metadata

## Error Handling

### Error Categories
1. **Compilation Errors**: Syntax, missing symbols
2. **Validation Errors**: Correctness failures
3. **Profiling Errors**: Runtime failures
4. **System Errors**: GPU unavailable, timeouts

### Error Propagation
- Errors captured in subprocess
- Propagated through JobState
- Returned in API response
- Logged for debugging

## Future Extensions

### Planned Features
1. **CUDA Strategy**: Direct CUDA kernel compilation
2. **Triton Strategy**: Triton kernel support
3. **Distributed Evaluation**: Multi-node support
4. **Compilation Cache**: Cross-request caching
5. **Performance Dashboard**: Real-time metrics

### Extension Points
1. New compilation strategies
2. Custom validation logic
3. Alternative profiling methods
4. Additional metrics collectors
5. Enhanced error recovery

## Best Practices

### For Contributors
1. Follow strategy pattern for new kernel types
2. Use subprocess for dangerous operations
3. Add comprehensive error handling
4. Include metrics collection
5. Write tests for each component

### For Users
1. Use typed API (KernelCode) for clarity
2. Enable device metrics for debugging
3. Monitor server stats endpoint
4. Check subprocess logs for errors
5. Use appropriate kernel types

## Conclusion

The V2 architecture provides a robust, extensible foundation for kernel evaluation. The strategy pattern enables easy addition of new kernel types, while subprocess isolation ensures server stability. This design balances flexibility, safety, and performance for production use.
