"""
Shared data models for CUDA Evaluation Server V2
Enhanced with user-definable input/output specifications
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time
import torch

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def _enum_val(x):
    return x.value if isinstance(x, Enum) else x

def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else x


# Kernel Type Enum
class KernelType(str, Enum):
    """Kernel implementation types"""
    TORCH = "torch"           # Pure PyTorch (reference models)
    CUDA = "cuda"            # Raw CUDA kernels
    TRITON = "triton"        # Triton kernels
    TORCH_CUDA = "torch_cuda"  # PyTorch with embedded CUDA (current generated code)


@dataclass
class TensorData:
    """Literal tensor payload sent over JSON (optional)"""
    # Raw storage buffer (row-major unless 'strides' given)
    data_b64: str                  # base64 of raw bytes
    # Required to reconstruct
    dtype: str                     # e.g. "float32"
    shape: List[int] = field(default_factory=list)
    compress: str = "none"         # "none" | "zlib"

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "data_b64": self.data_b64,
            "compress": self.compress,
            "shape": self.shape,
            "dtype": self.dtype,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorData":
        return cls(
            data_b64=d["data_b64"],
            compress=d.get("compress", "none"),
            shape=[int(x) for x in d.get("shape", [])],
            dtype=d.get("dtype", "float32")
        )
    

@dataclass
class TensorInit:
    """How to generate a tensor deterministically on the server (if no data payload)."""
    kind: str = "randn"            # "randn" | "zeros" | "ones" | "uniform" | "full" | "arange"
    seed: Optional[int] = None
    mean: Optional[float] = None   # randn
    std: Optional[float] = None    # randn
    low: Optional[float] = None    # uniform
    high: Optional[float] = None   # uniform
    fill_value: Optional[float] = None  # full
    start: Optional[float] = None  # arange
    step: Optional[float] = None   # arange

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "kind": self.kind,
            "seed": self.seed,
            "mean": self.mean,
            "std": self.std,
            "low": self.low,
            "high": self.high,
            "fill_value": self.fill_value,
            "start": self.start,
            "step": self.step,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorInit":
        return cls(
            kind=d.get("kind", "randn"),
            seed=d.get("seed"),
            mean=d.get("mean"),
            std=d.get("std"),
            low=d.get("low"),
            high=d.get("high"),
            fill_value=d.get("fill_value"),
            start=d.get("start"),
            step=d.get("step"),
        )


@dataclass
class TensorSpec:
    """Specification for a tensor argument or output"""
    shape: List[int] = field(default_factory=list) # required when init is used; can be omitted if data provided
    dtype: str = "float32"
    init: Optional[TensorInit] = None    # generate on server
    data: Optional[TensorData] = None    # OR literal payload (mutually exclusive with init)
    
    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "shape": [int(x) for x in self.shape],
            "dtype": self.dtype,
            "init": self.init.to_dict() if self.init else None,
            "data": self.data.to_dict() if self.data else None,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSpec":
        return cls(
            shape=[int(x) for x in d["shape"]],
            dtype=d.get("dtype", "float32"),
            init=TensorInit.from_dict(d["init"]) if d.get("init") else None,
            data=TensorData.from_dict(d["data"]) if d.get("data") else None,
        )

@dataclass
class LaunchDim:
    x: int = 1
    y: int = 1
    z: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"x": int(self.x), "y": int(self.y), "z": int(self.z)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LaunchDim":
        return cls(x=int(d.get("x", 1)), y=int(d.get("y", 1)), z=int(d.get("z", 1)))

@dataclass
class LaunchConfig:
    grid: Optional[LaunchDim] = None
    block: Optional[LaunchDim] = None          # CUDA-only
    shared_mem_bytes: Optional[int] = None     # CUDA-only
    num_warps: Optional[int] = None            # Triton-only
    num_stages: Optional[int] = None           # Triton-only

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "grid": self.grid.to_dict() if self.grid else None,
            "block": self.block.to_dict() if self.block else None,
            "shared_mem_bytes": self.shared_mem_bytes,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LaunchConfig":
        return cls(
            grid=LaunchDim.from_dict(d["grid"]) if d.get("grid") else None,
            block=LaunchDim.from_dict(d["block"]) if d.get("block") else None,
            shared_mem_bytes=int(d["shared_mem_bytes"]) if d.get("shared_mem_bytes") is not None else None,
            num_warps=int(d["num_warps"]) if d.get("num_warps") is not None else None,
            num_stages=int(d["num_stages"]) if d.get("num_stages") is not None else None,
        )

@dataclass
class ArgSpec:
    """Kernel argument specification"""
    name: str
    type: str  # "tensor","int","float","str","bool"
    value: Optional[Union[int, float, str, bool]] = None
    tensor_spec: Optional[TensorSpec] = None
    role: str = "input"            # "input" | "output" | "inout"
    is_meta: bool = False          # Triton constexpr/meta

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "tensor_spec": self.tensor_spec.to_dict() if self.tensor_spec else None,
            "role": self.role,
            "is_meta": self.is_meta,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArgSpec":
        return cls(
            name=d["name"],
            type=d["type"],
            value=d.get("value"),
            tensor_spec=TensorSpec.from_dict(d["tensor_spec"]) if d.get("tensor_spec") else None,
            role=d.get("role", "input"),
            is_meta=bool(d.get("is_meta", False)),
        )


@dataclass
class IOContract:
    args: List[ArgSpec]
    outputs: List[TensorSpec] = field(default_factory=list)
    launch: Optional[LaunchConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "args": [a.to_dict() for a in self.args],
            "outputs": [o.to_dict() for o in self.outputs] if self.outputs else [],
            "launch": self.launch.to_dict() if self.launch else None,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IOContract":
        return cls(
            args=[ArgSpec.from_dict(x) for x in d.get("args", [])],
            outputs=[TensorSpec.from_dict(x) for x in d.get("outputs", [])],
            launch=LaunchConfig.from_dict(d["launch"]) if d.get("launch") else None,
        )


# Kernel Code Wrapper
@dataclass
class KernelCode:
    """Wrapper for kernel source code with type information"""
    source_code: str
    kernel_type: KernelType
    io: Optional[IOContract] = None
    # Optional metadata for specific kernel types
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none({
            "source_code": self.source_code,
            "kernel_type": _enum_val(self.kernel_type),
            "metadata": self.metadata if self.metadata else None,
            "io": self.io.to_dict() if self.io else None,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelCode":
        kt = d.get("kernel_type")
        kt = KernelType(kt) if not isinstance(kt, KernelType) else kt
        return cls(
            source_code=d["source_code"],
            kernel_type=kt,
            metadata=d.get("metadata"),
            io=IOContract.from_dict(d["io"]) if d.get("io") else None,
        )


# Request/Response Models for HTTP API
class EvaluationRequest(BaseModel):
    """HTTP request model with explicit kernel types and optional input/output specs"""
    ref_kernel: KernelCode
    custom_kernel: KernelCode
    
    num_trials: int = 100
    timeout: int = 120
    
    class Config:
        """Pydantic config to handle dataclass fields"""
        arbitrary_types_allowed = True


class EvaluationResponse(BaseModel):
    """HTTP response model - maintains compatibility with existing API"""
    job_id: str
    kernel_exec_result: Dict
    ref_runtime: Dict
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
class ProfilingRequest:
    """Request for profiling service with kernel types"""
    ref_kernel: KernelCode
    custom_kernel: KernelCode
    num_trials: int = 100
    job_id: Optional[str] = None


@dataclass
class ProfilingResult:
    """Container for profiling results"""
    success: bool
    original_runtime: Optional[Dict[str, float]] = None
    custom_runtime: Optional[Dict[str, float]] = None
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
    request: EvaluationRequest
    created_at: float
    compilation_result: Optional[CompilationResult] = None
    validation_result: Optional[ValidationResult] = None
    profiling_result: Optional[ProfilingResult] = None
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
        self.output_specs = io_contract.outputs if io_contract and io_contract.outputs else None
        
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
        
    def _generate_inputs(self):
        """Generate inputs based on specs"""
        if self.input_specs:
            # Use ArgsGenerator to create inputs
            from shared.args_generator import ArgsGenerator
            generator = ArgsGenerator()
            return generator.generate_from_specs(self.input_specs, self.device)
        raise ValueError("No inputs provided and no input specs available")