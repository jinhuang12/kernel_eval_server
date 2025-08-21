"""
CUDA Evaluation Server V2 - Async CuPy-optimized version

This package provides a refactored version of the CUDA kernel evaluation server
with separated compilation and profiling services, FastAPI async endpoints,
and CuPy-only compilation for 5-20x speedup.

Components:
- Frontend: FastAPI app with async request handling
- Compilation Service: CuPy-based kernel compilation
- Profiling Service: GPU-based performance profiling  
- Job Manager: Workflow orchestration between services

Usage:
    python main.py --host 0.0.0.0 --port 8000
"""

__version__ = "2.0.0"
__author__ = "IronFist Team"
