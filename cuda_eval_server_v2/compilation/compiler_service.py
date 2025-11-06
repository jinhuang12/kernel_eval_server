"""
Compilation service with strategy pattern for different kernel types
Manages compilation strategies and delegates to appropriate strategy based on kernel type
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional

from compilation.base_compiler import BaseCompilationBackend
from compilation.torch_cuda.torch_cuda_backend import TorchCudaCompilationBackend
from compilation.torch.torch_backend import TorchCompilationBackend
from compilation.triton.triton_backend import TritonCompilationBackend
from compilation.cuda.cuda_backend import CudaCompilationBackend
from compilation.multi_kernel.multi_kernel_backend import MultiKernelCompilationBackend
from shared.models import BaseExecutableKernel, CompilationRequest, CompilationResult, KernelType
from shared.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class CompilationService:
    """
    Service for compiling kernels using strategy pattern
    Delegates compilation to appropriate strategy based on kernel type
    """
    
    def __init__(self, cache_dir: str = "/tmp/cupy_kernel_cache"):
        """
        Initialize compilation service with strategy registry
        
        Args:
            gpu_manager: Shared GPU resource manager
            cache_dir: Directory for compilation cache
        """
        self.cache_dir = cache_dir
        self.metrics_collector = get_metrics_collector()
        
        # Register compilation backends
        self.backends: Dict[KernelType, BaseCompilationBackend] = {
            KernelType.TORCH_CUDA: TorchCudaCompilationBackend(cache_dir),
            KernelType.TORCH: TorchCompilationBackend(),
            KernelType.TRITON: TritonCompilationBackend(),
            KernelType.CUDA: CudaCompilationBackend(cache_dir),
            KernelType.MULTI_KERNEL: MultiKernelCompilationBackend(),
        }
    
    def compile_kernel(self, request: CompilationRequest, gpu_id: Optional[int] = None) -> CompilationResult:
        """
        Compile kernel using appropriate strategy based on kernel type
        
        Args:
            request: CompilationRequest with KernelCode objects containing type info
            gpu_id: Optional GPU ID to use
            
        Returns:
            CompilationResult with compiled kernel or error
        """
        start_time = time.time()
    
        try:
            # Select strategy based on custom kernel type
            kernel_type = request.kernel_code.kernel_type
            strategy = self.backends.get(kernel_type)
            
            if not strategy:
                raise ValueError(f"No compilation strategy available for kernel type: {kernel_type}")
            
            # Compile using selected strategy
            kernel = strategy.compile(request.kernel_code, gpu_id)

            logger.info(f"Compiled kernel successfully using {strategy.__class__.__name__} for {kernel_type} kernel compilation")

            compilation_time = time.time() - start_time

            return CompilationResult(
                compiles=True,
                kernel=kernel,
                compilation_time=compilation_time
            )
            
        except Exception as e:
            compilation_time = time.time() - start_time
            logger.error(f"Compilation failed for job {request.job_id}: {e}")
            logger.error(f"Full stack trace: {traceback.format_exc()}")
            
            # Record failed compilation
            self.metrics_collector.record_compilation_end(
                request.job_id or "unknown",
                False,
                compilation_time,
                False
            )
            
            return CompilationResult(
                compiles=False,
                error=f"Compilation error:\n{traceback.format_exc()}",
                compilation_time=compilation_time
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for compilation service"""
        available_strategies = list(self.backends.keys())
        return {
            "status": "healthy",
            "cache_dir": self.cache_dir,
            "available_kernel_types": [str(kt) for kt in available_strategies],
            "strategies_loaded": len(self.backends)
        }
