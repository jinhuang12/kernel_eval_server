"""
Simplified compilation backend for pure CUDA kernels
Compiles CUDA kernels using CuPy RawModule for multi-kernel support
"""

import re
import logging
import torch
import cupy
from typing import List

from compilation.base_compiler import BaseCompilationBackend
from shared.models import KernelCode, KernelType
from shared.kernel_metadata import CudaKernelMetadata
from shared.executable_kernels import CudaExecutableKernel

logger = logging.getLogger(__name__)


class CudaCompilationBackend(BaseCompilationBackend):
    """
    CUDA kernel compilation backend using RawModule
    Supports multiple kernels in single source file
    """
    
    def __init__(self, cache_dir: str = "/tmp/cuda_kernel_cache"):
        self.cache_dir = cache_dir
    
    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> CudaExecutableKernel:
        """
        Compile CUDA source using RawModule for multi-kernel support
        
        Args:
            kernel: KernelCode with CUDA source and IOContract
            gpu_id: GPU device ID to compile on
            
        Returns:
            CudaExecutableKernel ready for execution
        """
        # Validate kernel type
        if kernel.kernel_type != KernelType.CUDA:
            raise ValueError(f"Expected CUDA kernel type, got {kernel.kernel_type}")
        
        # Get typed metadata
        metadata = kernel.get_typed_metadata()
        if not isinstance(metadata, CudaKernelMetadata):
            # Create from dict or use defaults
            metadata = CudaKernelMetadata.from_dict(
                kernel.metadata if isinstance(kernel.metadata, dict) else {}
            )
        
        device = torch.device(f'cuda:{gpu_id}')
        
        # Preprocess source (add extern "C" if needed)
        source = self._preprocess_source(kernel.source_code)
        
        # Extract kernel names to determine entrypoint
        kernel_names = self._extract_all_kernel_names(kernel.source_code)
        logger.info(f"Found kernels in source: {kernel_names}")
        
        # Compile with RawModule
        logger.info(f"Compiling CUDA module on GPU {gpu_id}")
        
        try:
            with cupy.cuda.Device(gpu_id):
                # Create RawModule with all kernels
                raw_module = cupy.RawModule(
                    code=source,
                    options=tuple(metadata.compiler_options) if metadata.compiler_options else (),
                    backend=metadata.backend,
                    name_expressions=tuple(metadata.name_expressions) if metadata.name_expressions else (),
                    jitify=metadata.jitify
                )
                logger.info(f"Successfully compiled CUDA module with {len(kernel_names)} kernel(s)")
        except Exception as e:
            logger.error(f"Failed to compile CUDA module: {e}")
            raise RuntimeError(f"CUDA compilation failed: {e}")
        
        # Determine entrypoint kernel
        entrypoint = metadata.kernel_name
        if not entrypoint:
            if len(kernel_names) == 1:
                # Single kernel - use it as default
                entrypoint = kernel_names[0]
                logger.info(f"Single kernel found, using as entrypoint: {entrypoint}")
            elif kernel_names:
                # Multiple kernels - use first as default
                entrypoint = kernel_names[0]
                logger.warning(f"Multiple kernels found without specified entrypoint, using first: {entrypoint}")
            else:
                raise RuntimeError("No kernels found in source")
        elif entrypoint not in kernel_names:
            logger.warning(f"Specified entrypoint '{entrypoint}' not found in kernel names {kernel_names}")
        
        # Return executable kernel with RawModule
        return CudaExecutableKernel(
            device=device,
            raw_module=raw_module,
            entrypoint_name=entrypoint,
            io_contract=kernel.io
        )
    
    def _extract_all_kernel_names(self, source: str) -> List[str]:
        """
        Extract all __global__ kernel function names from source
        
        Args:
            source: CUDA source code
            
        Returns:
            List of kernel function names
        """
        # Pattern to match __global__ function declaration
        pattern = r'__global__\s+(?:void|[\w:]+)\s+(\w+)\s*\('
        return re.findall(pattern, source)
    
    def _preprocess_source(self, source: str) -> str:
        """
        Minimal preprocessing - add extern "C" if needed for CuPy
        
        Args:
            source: Original CUDA source
            
        Returns:
            Preprocessed source
        """
        # Check if extern "C" already exists
        if 'extern "C"' in source:
            return source
        
        # Add extern "C" wrapper around __global__ functions
        # This is required for CuPy RawModule to find the functions
        pattern = r'(__global__\s+(?:void|[\w:]+)\s+\w+\s*\([^)]*\)\s*{)'
        
        def add_extern_c(match):
            return f'extern "C" {match.group(1)}'
        
        return re.sub(pattern, add_extern_c, source)