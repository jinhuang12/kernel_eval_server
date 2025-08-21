"""
Base compilation backend for kernel compilation
Provides abstract interface for different kernel type compilers with subprocess validation
"""

from abc import ABC, abstractmethod
import logging

from shared.models import BaseExecutableKernel, KernelCode
from typing import TypeVar

# TypeVar bound to BaseModel
T = TypeVar('T', bound=BaseExecutableKernel)

logger = logging.getLogger(__name__)


class BaseCompilationBackend(ABC):
    """Abstract base class for kernel compilation backend"""
    
    @abstractmethod
    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> BaseExecutableKernel:
        """
        Compile kernel and return executable kernel info
        
        Args:
            kernel: KernelCode with source and type information
            gpu_id: GPU device ID to compile on
            kw_args: keyword arguments

        Returns:
            Executable kernel 
        """
        pass
    
    def preprocess_kernel(self, kernel: KernelCode) -> KernelCode:
        """
        Optional preprocessing step for kernel code
        Override in subclasses if needed
        
        Args:
            kernel: Original kernel code
            
        Returns:
            Preprocessed kernel code
        """
        return kernel
