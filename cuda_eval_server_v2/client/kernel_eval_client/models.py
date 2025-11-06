"""
Client-specific models for Kernel Evaluation Client Library

Most models are imported from shared.models. This file contains only
client-specific additions like response parsing models.
"""

import sys
from pathlib import Path
from typing import Dict

# Add parent directories to path to import from shared
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all models from shared
from shared.models import (
    # Enums
    KernelType,
    # Tensor specifications
    TensorData,
    TensorInit,
    TensorSpec,
    # Launch configuration
    LaunchDim,
    LaunchConfig,
    # Arguments
    ArgSpec,
    IOContract,
    # Kernel
    KernelCode,
)

# Re-export all shared models
__all__ = [
    "KernelType",
    "TensorData", 
    "TensorInit",
    "TensorSpec",
    "LaunchDim",
    "LaunchConfig",
    "ArgSpec",
    "IOContract",
    "KernelCode",
    "RuntimeStats",  # Client-specific
]


# Client-specific models (for response parsing)
from dataclasses import dataclass


@dataclass
class RuntimeStats:
    """Runtime statistics from kernel profiling (client-side parsing)"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RuntimeStats":
        """Create from dictionary, handling missing fields"""
        return cls(
            mean=d.get("mean", 0.0),
            std=d.get("std", 0.0),
            min=d.get("min", 0.0),
            max=d.get("max", 0.0),
            median=d.get("median", 0.0),
            percentile_95=d.get("percentile_95", 0.0),
            percentile_99=d.get("percentile_99", 0.0)
        )