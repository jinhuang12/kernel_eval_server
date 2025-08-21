"""
Validation module for kernel correctness testing
Provides subprocess isolation and kernel-type specific validators
"""

# Use conditional imports for subprocess vs main process contexts
try:
    # Try relative imports first (for main process)
    from .base_validator import BaseKernelValidator
    from .correctness_validator import CorrectnessValidator
except ImportError:
    # Fall back to absolute imports (for subprocess)
    from validation.base_validator import BaseKernelValidator
    from validation.correctness_validator import CorrectnessValidator


__all__ = [
    'BaseKernelValidator',
    'CorrectnessValidator', 
    'TorchValidator',
]
