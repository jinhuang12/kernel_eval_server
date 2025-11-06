"""
Multi-kernel compilation backend for Python sequences with mixed kernel types
"""

import logging
import torch
from typing import Optional

from compilation.base_compiler import BaseCompilationBackend
from shared.models import BaseExecutableKernel, KernelCode, KernelType
from shared.executable_kernels import MultiKernelExecutableKernel
from shared.kernel_metadata import MultiKernelMetadata

logger = logging.getLogger(__name__)


class MultiKernelCompilationBackend(BaseCompilationBackend):
    """
    Compilation backend for multi-kernel Python sequences

    Validates:
    - Entry point function exists
    - IOContract is provided
    - Code can be loaded as module
    """

    def __init__(self):
        self.logger = logger

    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> BaseExecutableKernel:
        """
        Compile multi-kernel sequence into executable form

        Args:
            kernel: KernelCode with Python source, metadata, and IOContract
            gpu_id: GPU device ID

        Returns:
            MultiKernelExecutableKernel ready for execution
        """
        device = torch.device(f"cuda:{gpu_id}")

        # Validate kernel type
        if kernel.kernel_type != KernelType.MULTI_KERNEL:
            raise ValueError(f"Expected MULTI_KERNEL kernel type, got {kernel.kernel_type}")

        # Require IOContract
        if not kernel.io:
            raise ValueError(
                "IOContract is required for multi-kernel sequences. "
                "Please provide input/output specifications."
            )

        # Extract and validate metadata
        metadata = kernel.get_typed_metadata()
        if not metadata:
            if not kernel.metadata or not isinstance(kernel.metadata, dict):
                raise ValueError("Metadata with 'entry_point' is required")
            metadata = kernel.metadata

        if isinstance(metadata, dict):
            metadata = MultiKernelMetadata.from_dict(metadata)

        if not isinstance(metadata, MultiKernelMetadata):
            raise ValueError(f"Expected MultiKernelMetadata, got {type(metadata)}")

        logger.info(f"Compiling multi-kernel sequence with entry point: {metadata.entry_point}")

        # Build executable kernel
        # Validation of entry point happens in __init__
        return MultiKernelExecutableKernel(
            kernel_code=kernel,
            device=device,
            metadata=metadata
        )
