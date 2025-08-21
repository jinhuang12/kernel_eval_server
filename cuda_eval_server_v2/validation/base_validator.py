"""
Base validator interface for kernel correctness validation
Each kernel type implements its own validation strategy
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
# Use conditional import for subprocess vs main process contexts
try:
    # Try relative import first (for main process)
    from ..shared.models import BaseExecutableKernel, ValidationResult
except ImportError:
    # Fall back to absolute import (for subprocess)
    from shared.models import BaseExecutableKernel, ValidationResult


class BaseKernelValidator(ABC):
    """Abstract base class for kernel validators"""
    
    @abstractmethod
    def validate_correctness(
        self,
        ref_kernel: BaseExecutableKernel,
        custom_kernel: BaseExecutableKernel,
        device: torch.device,
        num_correct_trials: int = 1,
        job_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate kernel correctness against reference
        
        Args:
            ref_kernel: Reference kernel code
            custom_kernel: Custom kernel code to validate
            compiled_info: Compiled kernel information
            device: CUDA device to use
            num_correct_trials: Number of correctness trials
            job_id: Optional job ID for logging/metrics
            
        Returns:
            ValidationResult with validation outcome
        """
        pass
