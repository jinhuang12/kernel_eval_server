"""
Compilation strategy for TORCH_CUDA kernel type
Handles PyTorch models with embedded CUDA kernels (current generated code)
"""

import logging
import torch
from typing import Dict, Any, Optional, List

from compilation.base_compiler import BaseCompilationBackend
from shared.models import BaseExecutableKernel, CompiledKernelInfo, IOContract, KernelCode, KernelType
from shared.executable_kernels import TorchCudaExecutableKernel
from .kernel_extractor import TorchCudaExtractor
from .compiler import TorchCudaCupyCompiler

logger = logging.getLogger(__name__)


class TorchCudaCompilationBackend(BaseCompilationBackend):
    """
    Handles TORCH_CUDA kernels with extraction and transformation
    This is the complex path for PyTorch models with embedded CUDA
    """
    
    def __init__(self, cache_dir: str = "/tmp/cupy_kernel_cache"):
        self.extractor = TorchCudaExtractor()
        self.compiler = TorchCudaCupyCompiler(cache_dir=cache_dir)
    
    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> TorchCudaExecutableKernel:
        """
        Compile TORCH_CUDA kernel with extraction and transformation
        
        Args:
            kernel: KernelCode with TORCH_CUDA source
            gpu_id: GPU device ID to compile on
            
        Returns:
            TorchExecutableKernel
        """
        device = torch.device(f'cuda:{gpu_id}')
        compiled_info = self._get_compilation_info(kernel, gpu_id)
        # Pass IOContract if available
        return TorchCudaExecutableKernel(device, compiled_info, io_contract=kernel.io)

    
    def _get_compilation_info(self, kernel: KernelCode, gpu_id: int) -> CompiledKernelInfo:     
        # Step 1: Extract CUDA kernel from torch_cuda source
        logger.info("Extracting CUDA kernel from TORCH_CUDA source")
        cuda_kernel = self.extractor.extract(kernel.source_code)
        
        if not cuda_kernel:
            return CompiledKernelInfo(
                kernel_type=KernelType.TORCH_CUDA,
                kernel_name="unknown",
                compilation_successful=False,
                gpu_id=gpu_id,
                error="Failed to extract CUDA kernel from TORCH_CUDA source",
                original_kernel_source=kernel.source_code
            )
        
        # Step 2: Compile CUDA kernel with CuPy and C++ wrapper transformation
        logger.info(f"Compiling CUDA kernel with CuPy on GPU {gpu_id}")
        compiled_result = self.compiler.compile_cuda_kernel(cuda_kernel, gpu_id)
        
        if not compiled_result or not compiled_result.get("compilation_successful", False):
            error_msg = "CuPy compilation failed"
            if compiled_result and "error" in compiled_result:
                error_msg = compiled_result["error"]
            elif compiled_result and "compilation_errors" in compiled_result:
                error_msg = f"Compilation errors: {'; '.join(compiled_result['compilation_errors'])}"
            
            return CompiledKernelInfo(
                kernel_type=KernelType.TORCH_CUDA,
                kernel_name="unknown",
                compilation_successful=False,
                gpu_id=gpu_id,
                error=error_msg,
                original_kernel_source=kernel.source_code
            )
        
        # Create CompiledKernelInfo with recompiled data
        return CompiledKernelInfo(
            kernel_type=KernelType.TORCH_CUDA,
            kernel_name=compiled_result["kernel_name"],
            compilation_successful=True,
            gpu_id=gpu_id,
            compiled_functions=compiled_result.get("compiled_functions"),
            cpp_wrapper=compiled_result.get("cpp_wrapper"),
            model_new_source=compiled_result.get("model_new_source"),
            cuda_source=compiled_result.get("cuda_source"),
            original_cuda_source=compiled_result.get("original_cuda_source"),
            original_kernel_source=kernel.source_code
        )
    