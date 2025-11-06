"""
Shared data models for CUDA Evaluation Server V2
Enhanced with user-definable input/output specifications
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time
import torch

from shared.kernel_metadata import KernelMetadata, BaseKernelMetadata, CudaKernelMetadata, TorchKernelMetadata, TritonKernelMetadata, MultiKernelMetadata
from shared.constants import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_NUM_TRIALS, DEFAULT_TIMEOUT

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def _enum_val(x):
    return x.value if isinstance(x, Enum) else x

def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else x

def _round_float_values(obj: Any, decimal_places: int = 3) -> Any:
    """
    Recursively round all float values in nested dictionaries/lists to specified decimal places.

    Args:
        obj: Object to process (dict, list, float, or other)
        decimal_places: Number of decimal places to round to (default: 3)

    Returns:
        Object with all float values rounded
    """
    if isinstance(obj, dict):
        return {k: _round_float_values(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_float_values(item, decimal_places) for item in obj]
    elif isinstance(obj, float):
        return round(obj, decimal_places)
    else:
        return obj


# Kernel Type Enum
class KernelType(str, Enum):
    """Kernel implementation types"""
    TORCH = "torch"           # Pure PyTorch (reference models)
    CUDA = "cuda"            # Raw CUDA kernels
    TRITON = "triton"        # Triton kernels
    TORCH_CUDA = "torch_cuda"  # PyTorch with embedded CUDA (current generated code)
    MULTI_KERNEL = "multi_kernel"  # Python scripts with mixed kernel types


class TensorData(BaseModel):
    """Literal tensor payload sent over JSON (optional)"""
    # Raw storage buffer (row-major unless 'strides' given)
    data_b64: str  # base64 of raw bytes
    # Required to reconstruct
    dtype: str = "float32"  # e.g. "float32"
    shape: List[int] = Field(default_factory=list)
    compress: str = "none"  # "none" | "zlib"

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorData":
        """Backward compatibility wrapper"""
        return cls(**d)
    

class TensorInit(BaseModel):
    """How to generate a tensor deterministically on the server (if no data payload)."""
    kind: str = "randn"  # "randn" | "zeros" | "ones" | "uniform" | "full" | "arange"
    seed: Optional[int] = None
    mean: Optional[float] = None  # randn
    std: Optional[float] = None  # randn
    low: Optional[float] = None  # uniform
    high: Optional[float] = None  # uniform
    fill_value: Optional[float] = None  # full
    start: Optional[float] = None  # arange
    step: Optional[float] = None  # arange

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorInit":
        """Backward compatibility wrapper"""
        return cls(**d)


class TensorSpec(BaseModel):
    """Specification for a tensor argument or output"""
    shape: List[int] = Field(default_factory=list)  # required when init is used; can be omitted if data provided
    dtype: str = "float32"
    init: Optional[TensorInit] = None  # generate on server
    data: Optional[TensorData] = None  # OR literal payload (mutually exclusive with init)

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSpec":
        """Backward compatibility wrapper"""
        # Handle nested models that might be dicts
        if d.get("init") and isinstance(d["init"], dict):
            d["init"] = TensorInit(**d["init"])
        if d.get("data") and isinstance(d["data"], dict):
            d["data"] = TensorData(**d["data"])
        return cls(**d)

class LaunchDim(BaseModel):
    x: int = 1
    y: int = 1
    z: int = 1

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LaunchDim":
        """Backward compatibility wrapper"""
        return cls(**d)

class LaunchConfig(BaseModel):
    grid: Optional[LaunchDim] = None
    block: Optional[LaunchDim] = None  # CUDA-only
    num_warps: Optional[int] = None  # Triton-only
    num_stages: Optional[int] = None  # Triton-only

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LaunchConfig":
        """Backward compatibility wrapper"""
        # Handle nested models that might be dicts
        if d.get("grid") and isinstance(d["grid"], dict):
            d["grid"] = LaunchDim(**d["grid"])
        if d.get("block") and isinstance(d["block"], dict):
            d["block"] = LaunchDim(**d["block"])
        return cls(**d)

class ArgSpec(BaseModel):
    """Kernel argument specification"""
    name: str
    type: str  # "tensor","int","float","str","bool"
    value: Optional[Union[int, float, str, bool]] = None
    tensor_spec: Optional[TensorSpec] = None
    role: str = "input"  # "input" | "output" | "inout"
    is_meta: bool = False  # Triton constexpr/meta

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArgSpec":
        """Backward compatibility wrapper"""
        # Handle nested model that might be dict
        if d.get("tensor_spec") and isinstance(d["tensor_spec"], dict):
            d["tensor_spec"] = TensorSpec(**d["tensor_spec"])
        return cls(**d)


class IOContract(BaseModel):
    args: List[ArgSpec]
    launch: Optional[LaunchConfig] = None

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IOContract":
        """Backward compatibility wrapper"""
        # Handle nested models that might be dicts
        if d.get("args"):
            d["args"] = [ArgSpec(**arg) if isinstance(arg, dict) else arg for arg in d["args"]]
        if d.get("launch") and isinstance(d["launch"], dict):
            d["launch"] = LaunchConfig(**d["launch"])
        return cls(**d)


# Kernel Code Wrapper
class KernelCode(BaseModel):
    """Wrapper for kernel source code with type information"""
    source_code: str
    kernel_type: KernelType
    io: Optional[IOContract] = None
    # Metadata can be typed dataclass or dict for backward compatibility
    metadata: Optional[Union[BaseKernelMetadata, Dict[str, Any]]] = None

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    @field_validator('metadata', mode='before')
    @classmethod
    def parse_metadata(cls, v, values):
        """Parse metadata if it's a dict and convert to appropriate type if possible"""
        # If already a BaseKernelMetadata instance, keep it
        if isinstance(v, BaseKernelMetadata):
            return v
        # Otherwise keep as dict for backward compatibility
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        result = self.model_dump(exclude_none=True)
        # Handle metadata specially for backward compatibility
        if self.metadata:
            if isinstance(self.metadata, BaseKernelMetadata):
                result["metadata"] = self.metadata.model_dump()
            else:
                result["metadata"] = self.metadata
        # Convert enum to string
        result["kernel_type"] = _enum_val(self.kernel_type)
        # Handle io specially if needed
        if self.io and hasattr(self.io, 'to_dict'):
            result["io"] = self.io.to_dict()
        return result

    def get_typed_metadata(self) -> Optional[BaseKernelMetadata]:
        """Get metadata as typed dataclass if possible"""
        if isinstance(self.metadata, BaseKernelMetadata):
            # Check if it's the correct subtype for this kernel_type
            # If it's a generic BaseKernelMetadata, convert it to the proper type
            if type(self.metadata) == BaseKernelMetadata:
                # Convert to dict and then to proper type
                metadata_dict = self.metadata.model_dump()
                if self.kernel_type == KernelType.CUDA:
                    return CudaKernelMetadata.from_dict(metadata_dict)
                elif self.kernel_type == KernelType.TORCH:
                    return TorchKernelMetadata.from_dict(metadata_dict)
                elif self.kernel_type == KernelType.TRITON:
                    return TritonKernelMetadata.from_dict(metadata_dict)
                elif self.kernel_type == KernelType.MULTI_KERNEL:
                    return MultiKernelMetadata.from_dict(metadata_dict)
            # Already correct type, return as-is
            return self.metadata
        elif isinstance(self.metadata, dict):
            if self.kernel_type == KernelType.CUDA:
                return CudaKernelMetadata.from_dict(self.metadata)
            elif self.kernel_type == KernelType.TORCH:
                return TorchKernelMetadata.from_dict(self.metadata)
            elif self.kernel_type == KernelType.TRITON:
                return TritonKernelMetadata.from_dict(self.metadata)
            elif self.kernel_type == KernelType.MULTI_KERNEL:
                return MultiKernelMetadata.from_dict(self.metadata)
        return None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelCode":
        """Backward compatibility wrapper"""
        kt = d.get("kernel_type")
        kt = KernelType(kt) if not isinstance(kt, KernelType) else kt
        # Handle nested models that might be dicts
        if d.get("io") and isinstance(d["io"], dict):
            d["io"] = IOContract(**d["io"])
        return cls(
            source_code=d["source_code"],
            kernel_type=kt,
            metadata=d.get("metadata"),
            io=d.get("io")
        )


# Response Data Models
class RuntimeStats(BaseModel):
    """Runtime statistics from kernel profiling"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    percentile_95: float = 0.0
    percentile_99: float = 0.0

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RuntimeStats":
        """Backward compatibility wrapper - Create from dictionary, handling missing fields"""
        return cls(
            mean=d.get("mean", 0.0),
            std=d.get("std", 0.0),
            min=d.get("min", 0.0),
            max=d.get("max", 0.0),
            median=d.get("median", 0.0),
            percentile_95=d.get("percentile_95", 0.0),
            percentile_99=d.get("percentile_99", 0.0)
        )


class SpeedOfLightMetrics(BaseModel):
    """Speed of Light performance metrics"""
    compute_memory_throughput_pct: Optional[float] = None
    compute_throughput_pct: Optional[float] = None
    sm_throughput_pct: Optional[float] = None
    gpu_dram_throughput_pct: Optional[float] = None
    memory_throughput_pct: Optional[float] = None
    dram_throughput_pct: Optional[float] = None

    model_config = ConfigDict(extra='allow')


class DetailedMetrics(BaseModel):
    """Detailed performance metrics"""
    l1_hit_rate_pct: Optional[float] = None
    l2_hit_rate_pct: Optional[float] = None
    warp_occupancy_pct: Optional[float] = None
    sm_active_cycles_pct: Optional[float] = None
    instructions_per_cycle: Optional[float] = None
    waves_per_sm: Optional[float] = None

    model_config = ConfigDict(extra='allow')


class MemoryMetrics(BaseModel):
    """Memory hierarchy performance metrics"""
    dram_avg_bandwidth_gb_s: Optional[float] = None
    dram_total_bandwidth_gb_s: Optional[float] = None
    dram_active_cycles_pct: Optional[float] = None
    l1_writeback_active_pct: Optional[float] = None
    l1_read_sectors_pct: Optional[float] = None
    l2_throughput_pct: Optional[float] = None

    model_config = ConfigDict(extra='allow')


class ComputeMetrics(BaseModel):
    """Compute utilization metrics"""
    fma_pipe_utilization_pct: Optional[float] = None
    fp64_pipe_utilization_pct: Optional[float] = None
    alu_pipe_utilization_pct: Optional[float] = None
    xu_pipe_utilization_pct: Optional[float] = None
    tensor_pipe_utilization_pct: Optional[float] = None
    instructions_per_cycle: Optional[float] = None
    occupancy_limit_blocks: Optional[float] = None
    occupancy_limit_registers: Optional[float] = None
    occupancy_limit_shared_mem: Optional[float] = None
    occupancy_limit_warps: Optional[float] = None
    registers_per_thread: Optional[float] = None

    model_config = ConfigDict(extra='allow')


class PipelineMetrics(BaseModel):
    """Pipeline utilization metrics"""
    fma_pipe_active_pct: Optional[float] = None
    alu_pipe_active_pct: Optional[float] = None
    tensor_pipe_active_pct: Optional[float] = None
    shared_pipe_active_pct: Optional[float] = None
    fp64_pipe_active_pct: Optional[float] = None
    sm_issue_active_pct: Optional[float] = None

    model_config = ConfigDict(extra='allow')


class OccupancyMetrics(BaseModel):
    """Occupancy and launch configuration metrics"""
    occupancy_limit_registers: Optional[float] = None
    occupancy_limit_shared_mem: Optional[float] = None
    occupancy_limit_warps: Optional[float] = None
    occupancy_limit_blocks: Optional[float] = None
    waves_per_sm: Optional[float] = None  # Changed from waves_per_multiprocessor to match parser
    block_size: Optional[float] = None
    grid_size: Optional[float] = None
    shared_mem_per_block: Optional[float] = None
    registers_per_thread: Optional[float] = None
    thread_occupancy_pct: Optional[float] = None
    block_occupancy_pct: Optional[float] = None
    warp_occupancy_pct: Optional[float] = None

    model_config = ConfigDict(extra='allow')


class ComparisonDeviceMetrics(BaseModel):
    """Device metrics from NCU profiling for kernel comparison (original vs custom)"""
    original_device_metrics: Optional['DeviceMetrics'] = None
    custom_device_metrics: Optional['DeviceMetrics'] = None

    model_config = ConfigDict(extra='allow')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper - Convert to dictionary maintaining the comparison structure expected by tests"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComparisonDeviceMetrics":
        """Backward compatibility wrapper - Create ComparisonDeviceMetrics from dictionary returned by parser for comparisons"""
        # Handle nested models that might be dicts
        if d.get("original_device_metrics") and isinstance(d["original_device_metrics"], dict):
            d["original_device_metrics"] = DeviceMetrics(**d["original_device_metrics"])
        if d.get("custom_device_metrics") and isinstance(d["custom_device_metrics"], dict):
            d["custom_device_metrics"] = DeviceMetrics(**d["custom_device_metrics"])
        return cls(**d)


class DeviceMetrics(BaseModel):
    """Device metrics from NCU profiling, organized by category"""
    speed_of_light: Optional[SpeedOfLightMetrics] = None
    detailed_metrics: Optional[DetailedMetrics] = None
    memory_metrics: Optional[MemoryMetrics] = None
    compute_metrics: Optional[ComputeMetrics] = None
    pipeline_metrics: Optional[PipelineMetrics] = None
    occupancy_metrics: Optional[OccupancyMetrics] = None

    model_config = ConfigDict(extra='allow')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        result = self.model_dump(exclude_none=True)
        # Round all float values to 3 decimal places
        return _round_float_values(result, decimal_places=3)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeviceMetrics":
        """Backward compatibility wrapper - Create DeviceMetrics from dictionary returned by parser"""
        # Handle nested models that might be dicts
        if d.get("speed_of_light") and isinstance(d["speed_of_light"], dict):
            d["speed_of_light"] = SpeedOfLightMetrics(**d["speed_of_light"])
        if d.get("detailed_metrics") and isinstance(d["detailed_metrics"], dict):
            d["detailed_metrics"] = DetailedMetrics(**d["detailed_metrics"])
        if d.get("memory_metrics") and isinstance(d["memory_metrics"], dict):
            d["memory_metrics"] = MemoryMetrics(**d["memory_metrics"])
        if d.get("compute_metrics") and isinstance(d["compute_metrics"], dict):
            d["compute_metrics"] = ComputeMetrics(**d["compute_metrics"])
        if d.get("pipeline_metrics") and isinstance(d["pipeline_metrics"], dict):
            d["pipeline_metrics"] = PipelineMetrics(**d["pipeline_metrics"])
        if d.get("occupancy_metrics") and isinstance(d["occupancy_metrics"], dict):
            d["occupancy_metrics"] = OccupancyMetrics(**d["occupancy_metrics"])
        return cls(**d)


class KernelMetadata(BaseModel):
    """Metadata about kernel execution"""
    gpu_id: int
    device_metrics: Optional[Union[DeviceMetrics, ComparisonDeviceMetrics]] = None
    kernel_name: Optional[str] = None
    kernel_type: Optional[str] = None
    gpu_type: Optional[str] = None
    cuda_version: Optional[str] = None

    model_config = ConfigDict(extra='allow')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)


class KernelExecutionResult(BaseModel):
    """Result of kernel execution including compilation, validation, and profiling"""
    compiled: bool
    correctness: bool
    runtime: float  # Mean runtime in ms
    metadata: KernelMetadata
    runtime_stats: Optional[RuntimeStats] = None
    compilation_error: Optional[str] = None
    validation_error: Optional[str] = None

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)


# Request/Response Models for HTTP API - Comparison (two kernels)
class CompareRequest(BaseModel):
    """HTTP request model for comparing two kernels"""
    ref_kernel: KernelCode
    custom_kernel: KernelCode

    num_trials: int = DEFAULT_NUM_TRIALS
    timeout: int = DEFAULT_TIMEOUT
    atol: Optional[float] = DEFAULT_ATOL
    rtol: Optional[float] = DEFAULT_RTOL
    
    @field_validator('ref_kernel', 'custom_kernel', mode='before')
    @classmethod
    def parse_kernel_code(cls, v):
        """Parse KernelCode from dict if needed"""
        if isinstance(v, dict):
            return KernelCode(**v)
        return v


class CompareResponse(BaseModel):
    """HTTP response model for kernel comparison"""
    job_id: str
    kernel_exec_result: KernelExecutionResult
    ref_runtime: RuntimeStats
    pod_name: str
    pod_ip: str
    status: str


# Request/Response Models for HTTP API - Single Kernel Evaluation
class EvaluationRequest(BaseModel):
    """HTTP request model for evaluating a single kernel"""
    kernel: KernelCode

    num_trials: int = DEFAULT_NUM_TRIALS
    timeout: int = DEFAULT_TIMEOUT

    @field_validator('kernel', mode='before')
    @classmethod
    def parse_kernel_code(cls, v):
        """Parse KernelCode from dict if needed"""
        if isinstance(v, dict):
            return KernelCode(**v)
        return v


class EvaluationResponse(BaseModel):
    """HTTP response model for single kernel evaluation"""
    job_id: str
    kernel_exec_result: KernelExecutionResult
    pod_name: str
    pod_ip: str
    status: str


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    compilation_service: Dict[str, Any]
    profiling_service: Dict[str, Any]


# Internal Service Models
@dataclass
class CompiledKernelInfo:
    """Structured compilation result with kernel-type specific fields"""
    kernel_type: KernelType
    kernel_name: str
    compilation_successful: bool
    
    # Common fields
    gpu_id: int
    
    # TORCH_CUDA specific
    compiled_functions: Optional[Dict[str, Any]] = None  # CuPy RawKernel objects
    cpp_wrapper: Optional[Dict[str, Any]] = None  # Compiled C++ extension
    model_new_source: Optional[str] = None
    cuda_source: Optional[str] = None  # Preprocessed CUDA source
    original_cuda_source: Optional[str] = None
    original_kernel_source: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    compilation_errors: Optional[List[str]] = None


@dataclass
class CompilationRequest:
    """Request for compilation service with kernel type info"""
    kernel_code: KernelCode
    job_id: Optional[str] = None


@dataclass 
class CompilationResult:
    """Result from compilation service"""
    compiles: bool
    kernel: Optional['BaseExecutableKernel'] = None
    compilation_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class CompareProfilingRequest:
    """Request for profiling service to compare two kernels"""
    ref_kernel: KernelCode
    custom_kernel: KernelCode
    num_trials: int = 100
    job_id: Optional[str] = None


@dataclass
class CompareProfilingResult:
    """Container for comparison profiling results"""
    success: bool
    original_runtime: Optional[Dict[str, float]] = None
    custom_runtime: Optional[Dict[str, float]] = None
    speedup: float = 0.0
    error: Optional[str] = None
    gpu_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProfilingRequest:
    """Request for profiling a single kernel"""
    kernel: KernelCode
    num_trials: int = 100
    job_id: Optional[str] = None


@dataclass
class ProfilingResult:
    """Container for single kernel profiling results"""
    success: bool
    runtime_stats: Dict[str, float]
    error: Optional[str] = None
    gpu_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


# Validation Result Models
@dataclass
class ValidationResult:
    """Result from subprocess validation"""
    is_correct: bool
    trials_passed: int = 0
    total_trials: int = 0
    max_difference: Optional[float] = None
    avg_difference: Optional[float] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


# Job Management Models
@dataclass
class JobState:
    """Internal job state tracking"""
    job_id: str
    status: str  # submitted, compiling, profiling, completed, failed
    request: Union[EvaluationRequest, CompareRequest]
    created_at: float
    compilation_result: Optional[CompilationResult] = None
    validation_result: Optional[ValidationResult] = None
    profiling_result: Optional[Union[ProfilingResult, CompareProfilingResult]] = None
    result: Optional[EvaluationResponse] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class BaseExecutableKernel(ABC):
    """Context-aware executable kernel that can be passed around and executed"""
    
    def __init__(self, 
                 kernel_type: KernelType,
                 device: torch.device,
                 io_contract: Optional[IOContract] = None):
        """
        Initialize executable kernel
        
        Args:
            kernel_type: Type of kernel (TORCH, CUDA, TORCH_CUDA, etc.)
            device: CUDA device to execute on
            io_contract: Optional I/O specifications
        """
        self.kernel_type = kernel_type
        self.device = device
        self.input_specs = io_contract.args if io_contract and io_contract.args else None
        
        # Execution context
        self._default_inputs = None
        self._profiling_mode = False
        self._use_cuda_graphs = False
        
        # Initialize kernel-specific components
        self._initialize_kernel()
    
    @abstractmethod
    def _initialize_kernel(self):
        """Initialize kernel-specific components"""
        pass
        
    def with_inputs(self, *inputs):
        """Set default inputs (fluent interface)"""
        self._default_inputs = inputs
        return self
        
    def with_profiling(self, use_cuda_graphs=False):
        """Enable profiling mode"""
        self._profiling_mode = True
        self._use_cuda_graphs = use_cuda_graphs
        return self
        
    def __call__(self, *inputs):
        """Make the kernel directly callable"""
        return self._execute_impl(*inputs)
        
    @abstractmethod
    def _execute_impl(self, *inputs) -> Optional[Any]:
        """Actual execution implementation"""
        pass