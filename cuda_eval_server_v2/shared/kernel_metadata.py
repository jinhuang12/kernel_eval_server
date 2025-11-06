"""
Metadata models for different kernel types
Provides typed metadata with backward compatibility for dict-based metadata
"""

from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel


# Base metadata class that all kernel types can extend
class BaseKernelMetadata(BaseModel):
    """Base metadata for all kernel types"""
    
    class Config:
        """Pydantic config"""
        extra = "allow"  # Allow extra fields for flexibility


class CudaKernelMetadata(BaseKernelMetadata):
    """Metadata specific to CUDA kernels"""
    kernel_name: Optional[str] = None  # Entrypoint kernel to execute
    compiler_options: Optional[List[str]] = None  # e.g., ["-O3", "-use_fast_math"]
    backend: str = "nvrtc"  # "nvcc" or "nvrtc"
    name_expressions: Optional[List[str]] = None  # For template kernels
    jitify: bool = False  # Enable jitify for C++ features
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CudaKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            return cls()
        return cls(**d)


class TorchKernelMetadata(BaseKernelMetadata):
    """Metadata specific to PyTorch kernels"""
    function_name: Optional[str] = None  # Target standalone function to execute
    class_name: Optional[str] = None  # Target class name (default: "Model")
    method_name: Optional[str] = None  # Target method name (default: "forward")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TorchKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            return cls()
        return cls(**d)


class TritonKernelMetadata(BaseKernelMetadata):
    """Metadata specific to Triton kernels"""
    kernel_name: Optional[str] = None  # Entrypoint kernel to execute

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TritonKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            return cls()
        return cls(**d)


class MultiKernelMetadata(BaseKernelMetadata):
    """Metadata specific to multi-kernel Python sequences"""
    entry_point: str  # Function name to call (required)
    description: Optional[str] = None  # Optional description of what the sequence does

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            raise ValueError("MultiKernelMetadata requires 'entry_point' field")
        return cls(**d)


# Union type for metadata - can be Pydantic model or dict for flexibility
KernelMetadata = Union[BaseKernelMetadata, Dict[str, Any]]