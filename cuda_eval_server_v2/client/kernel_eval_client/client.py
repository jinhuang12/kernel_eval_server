"""
HTTP Client wrapper for Kernel Evaluation Server API
"""

import requests
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Union

# Add parent directories to path to import from shared
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import KernelCode
from .models import RuntimeStats  # Client-specific model


class KernelEvalClient:
    """
    Client for interacting with the Kernel Evaluation Server API.
    
    Example:
        client = KernelEvalClient("http://localhost:8000")
        
        # Compare two kernels
        result = client.compare(ref_kernel, custom_kernel, num_trials=100)
        
        # Evaluate single kernel
        result = client.evaluate(kernel, num_trials=100)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        """
        Initialize the client.
        
        Args:
            base_url: Server URL (default: http://localhost:8000)
            timeout: Request timeout in seconds (default: 120)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def compare(self, 
                ref_kernel: Union[KernelCode, Dict],
                custom_kernel: Union[KernelCode, Dict],
                num_trials: int = 100,
                timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare two kernel implementations.
        
        Args:
            ref_kernel: Reference kernel (KernelCode object or dict)
            custom_kernel: Custom kernel to compare (KernelCode object or dict)
            num_trials: Number of profiling trials (default: 100)
            timeout: Override default timeout for this request
            
        Returns:
            Response dictionary with comparison results
            
        Example:
            ref = KernelCode(
                source_code="class Model(torch.nn.Module):\\n    def forward(self, x): return x * 2",
                kernel_type=KernelType.TORCH
            )
            custom = KernelCode(
                source_code="# Optimized implementation",
                kernel_type=KernelType.TORCH_CUDA
            )
            result = client.compare(ref, custom, num_trials=200)
        """
        # Convert to dict if needed
        ref_dict = ref_kernel.to_dict() if hasattr(ref_kernel, 'to_dict') else ref_kernel
        custom_dict = custom_kernel.to_dict() if hasattr(custom_kernel, 'to_dict') else custom_kernel
        
        request_data = {
            "ref_kernel": ref_dict,
            "custom_kernel": custom_dict,
            "num_trials": num_trials,
            "timeout": timeout or self.timeout
        }
        
        response = self.session.post(
            f"{self.base_url}/compare",
            json=request_data,
            timeout=timeout or self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Parse runtime stats if present
        if "ref_runtime" in result and isinstance(result["ref_runtime"], dict):
            result["ref_runtime"] = RuntimeStats.from_dict(result["ref_runtime"])
        
        if "kernel_exec_result" in result:
            exec_result = result["kernel_exec_result"]
            if "runtime_stats" in exec_result and isinstance(exec_result["runtime_stats"], dict):
                exec_result["runtime_stats"] = RuntimeStats.from_dict(exec_result["runtime_stats"])
        
        return result
    
    def evaluate(self,
                 kernel: Union[KernelCode, Dict],
                 num_trials: int = 100,
                 timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a single kernel's performance.
        
        Args:
            kernel: Kernel to evaluate (KernelCode object or dict)
            num_trials: Number of profiling trials (default: 100)
            timeout: Override default timeout for this request
            
        Returns:
            Response dictionary with evaluation results
            
        Example:
            kernel = KernelCode(
                source_code=triton_code,
                kernel_type=KernelType.TRITON,
                io=io_contract
            )
            result = client.evaluate(kernel, num_trials=100)
        """
        # Convert to dict if needed
        kernel_dict = kernel.to_dict() if hasattr(kernel, 'to_dict') else kernel
        
        request_data = {
            "kernel": kernel_dict,
            "num_trials": num_trials,
            "timeout": timeout or self.timeout
        }
        
        response = self.session.post(
            f"{self.base_url}/evaluate",
            json=request_data,
            timeout=timeout or self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Parse runtime stats if present
        if "kernel_exec_result" in result:
            exec_result = result["kernel_exec_result"]
            if "runtime_stats" in exec_result and isinstance(exec_result["runtime_stats"], dict):
                exec_result["runtime_stats"] = RuntimeStats.from_dict(exec_result["runtime_stats"])
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.
        
        Returns:
            Health status dictionary
        """
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job.
        
        Args:
            job_id: Job UUID
            
        Returns:
            Job status dictionary
        """
        response = self.session.get(f"{self.base_url}/job/{job_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Server statistics dictionary
        """
        response = self.session.get(f"{self.base_url}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def cleanup_jobs(self) -> Dict[str, Any]:
        """
        Trigger manual cleanup of old jobs (admin endpoint).
        
        Returns:
            Cleanup result dictionary
        """
        response = self.session.post(f"{self.base_url}/admin/cleanup-jobs", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()