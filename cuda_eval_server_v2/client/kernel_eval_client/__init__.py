"""
Kernel Evaluation Client Library

A Python client library for interacting with the Kernel Evaluation Server API.
Provides utilities for building IOContracts and submitting kernel evaluation requests.
"""

from .models import (
    # Core models
    TensorSpec,
    TensorInit,
    TensorData,
    ArgSpec,
    IOContract,
    LaunchConfig,
    LaunchDim,
    KernelCode,
    KernelType,
)

from .builder import IOContractBuilder

from .specs import (
    # Tensor creation helpers
    create_tensor_spec,
    create_randn_spec,
    create_uniform_spec,
    create_zeros_spec,
    create_ones_spec,
    create_full_spec,
    create_arange_spec,
    # JSON utilities
    to_json,
    save_to_file,
    load_from_file,
)

from .client import KernelEvalClient

__version__ = "0.1.0"

__all__ = [
    # Models
    "TensorSpec",
    "TensorInit",
    "TensorData",
    "ArgSpec",
    "IOContract",
    "LaunchConfig",
    "LaunchDim",
    "KernelCode",
    "KernelType",
    # Builder
    "IOContractBuilder",
    # Spec helpers
    "create_tensor_spec",
    "create_randn_spec",
    "create_uniform_spec",
    "create_zeros_spec",
    "create_ones_spec",
    "create_full_spec",
    "create_arange_spec",
    # JSON utilities
    "to_json",
    "save_to_file",
    "load_from_file",
    # Client
    "KernelEvalClient",
]