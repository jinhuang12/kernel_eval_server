"""
Triton compilation strategy for Triton kernel evaluation
"""

from .triton_backend import TritonCompilationBackend, TritonExecutableKernel

__all__ = [
    'TritonCompilationBackend',
    'TritonExecutableKernel'
]
