"""
TORCH compilation strategy package
Handles pure PyTorch model compilation (no embedded CUDA)
"""

from .torch_backend import TorchCompilationBackend

__all__ = ['TorchCompilationBackend']
