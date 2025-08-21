"""
Compilation strategy for pure PyTorch models
Handles PyTorch models without embedded CUDA kernels
"""

import logging
import torch
from typing import Dict, Any, Optional, List

from compilation.base_compiler import BaseCompilationBackend
from shared.models import KernelCode, KernelType
from shared.executable_kernels import TorchExecutableKernel

logger = logging.getLogger(__name__)


class TorchCompilationBackend(BaseCompilationBackend):
    """
    Handles pure PyTorch models without embedded CUDA
    No compilation needed - just validation
    """
    def __init__(self):
       pass

    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> TorchExecutableKernel:
        """
        "Compile" PyTorch model - no actual compilation needed
        
        Args:
            kernel: KernelCode with PyTorch source
            gpu_id: GPU device ID (for consistency)
            
        Returns:
            Dictionary indicating PyTorch model is ready for execution
        """
        if kernel.kernel_type != KernelType.TORCH:
            raise ValueError(f"Invalid kernel for {self.__class__.__name__}")
        
        device = torch.device(f'cuda:{gpu_id}')
        return TorchExecutableKernel(device, kernel)
        
        
    
