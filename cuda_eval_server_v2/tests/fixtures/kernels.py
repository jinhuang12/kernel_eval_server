"""
Kernel library that loads actual test cases from test_data/ directory
No hardcoded kernels - all kernels come from JSON test files
"""

from typing import Dict, Any, Optional
from .test_data_loader import get_loader

class KernelLibrary:
    """Library of test kernels loaded from test_data/ directory"""
    
    # Mapping from method names to test case files and kernel types
    KERNEL_MAPPINGS = {
        "torch_add": ("triton_torch_triton_add", "ref_kernel"),
        "torch_matmul": ("cuda_matrix_multiply", "ref_kernel"),
        "torch_gelu": ("cuda_gelu_activation", "ref_kernel"),
        "triton_add": ("triton_torch_triton_add", "custom_kernel"),
        "triton_matmul": ("triton_triton_matmul_optimized", "custom_kernel"),
        "cuda_vector_add": ("cuda_vector_add", "custom_kernel"),
        "cuda_matrix_multiply": ("cuda_matrix_multiply", "custom_kernel"),
        "torch_cuda_add": ("cuda_vector_add", "custom_kernel"),  # Map to CUDA vector add
    }
    
    @staticmethod
    def _add_torch_metadata(kernel: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata for torch kernels with IOContract if not present"""
        if kernel and kernel.get("kernel_type") == "torch" and kernel.get("io") and "metadata" not in kernel:
            kernel["metadata"] = {"class_name": "Model", "method_name": "forward"}
        return kernel
    
    # ============= TORCH KERNELS =============
    @staticmethod
    def torch_add() -> Dict[str, Any]:
        """Simple torch.add reference kernel from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("triton_torch_triton_add")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "ref_kernel")
            return KernelLibrary._add_torch_metadata(kernel)
        return None
    
    @staticmethod
    def torch_matmul() -> Dict[str, Any]:
        """Torch matrix multiplication from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("cuda_matrix_multiply")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "ref_kernel")
            return KernelLibrary._add_torch_metadata(kernel)
        return None
    
    @staticmethod
    def torch_gelu() -> Dict[str, Any]:
        """GELU activation function from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("cuda_gelu_activation")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "ref_kernel")
            return KernelLibrary._add_torch_metadata(kernel)
        return None
    
    # ============= CUDA KERNELS =============
    @staticmethod
    def cuda_vector_add() -> Dict[str, Any]:
        """CUDA vector addition kernel from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("cuda_vector_add")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "custom_kernel")
            if kernel:
                return kernel
        return None
    
    @staticmethod
    def cuda_matrix_multiply() -> Dict[str, Any]:
        """CUDA matrix multiplication kernel from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("cuda_matrix_multiply")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "custom_kernel")
            if kernel:
                return kernel
        return None
    
    # ============= TORCH_CUDA KERNELS =============
    @staticmethod
    def torch_cuda_add() -> Dict[str, Any]:
        """PyTorch with embedded CUDA kernel - maps to CUDA vector add"""
        loader = get_loader()
        # Map to CUDA vector add test case
        test_case = loader.get_test_case("cuda_vector_add")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "custom_kernel")
            if kernel:
                return kernel
        return None
    
    # ============= TRITON KERNELS =============
    @staticmethod
    def triton_add() -> Dict[str, Any]:
        """Triton vector addition kernel from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("triton_torch_triton_add")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "custom_kernel")
            if kernel:
                return kernel
        return None
    
    @staticmethod
    def triton_matmul() -> Dict[str, Any]:
        """Optimized Triton matrix multiplication from test data"""
        loader = get_loader()
        test_case = loader.get_test_case("triton_triton_matmul_optimized")
        if test_case:
            kernel = loader.get_kernel_from_test_case(test_case, "custom_kernel")
            if kernel:
                return kernel
        return None
    
    # ============= ERROR KERNELS =============
    # These are kept as minimal inline definitions for testing error conditions
    @staticmethod
    def compilation_error() -> Dict[str, Any]:
        """Kernel that fails compilation - for testing error handling"""
        return {
            "kernel_type": "torch",
            "source_code": """
import torch

def get_inputs():
    x = torch.randn(1024, device='cuda')
    return [x]

def get_init_inputs():
    return []

def kernel_fn(x):
    # Syntax error for testing
    return torch.add(x, 
""",
            "io": None,
            "metadata": {"function_name": "kernel_fn"}
        }
    
    @staticmethod
    def runtime_error() -> Dict[str, Any]:
        """Kernel that crashes at runtime - for testing error handling"""
        return {
            "kernel_type": "torch",
            "source_code": """
import torch

def get_inputs():
    x = torch.randn(1024, device='cuda')
    return [x]

def get_init_inputs():
    return []

def kernel_fn(x):
    # Runtime error for testing
    import ctypes
    ctypes.string_at(0)
    return x
""",
            "io": None,
            "metadata": {"function_name": "kernel_fn"}
        }
    
    @staticmethod
    def validation_failure() -> Dict[str, Any]:
        """Kernel that returns incorrect results - for testing validation"""
        return {
            "kernel_type": "torch",
            "source_code": """
import torch

def get_inputs():
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    return [x, y]

def get_init_inputs():
    return []

def kernel_fn(x, y):
    # Returns wrong result for testing validation
    return torch.sub(x, y)  # Should be add
""",
            "io": None,
            "metadata": {"function_name": "kernel_fn"}
        }
    
    @classmethod
    def get_kernel(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get kernel by name - loads from test data or returns error kernels"""
        # Try to load from test data first using mapping
        if name in cls.KERNEL_MAPPINGS:
            test_case_name, kernel_type = cls.KERNEL_MAPPINGS[name]
            loader = get_loader()
            test_case = loader.get_test_case(test_case_name)
            if test_case:
                kernel = loader.get_kernel_from_test_case(test_case, kernel_type)
                if kernel:
                    # Add metadata for torch kernels with IOContract
                    return cls._add_torch_metadata(kernel)
        
        # Fall back to method lookup for error kernels
        kernels = {
            "torch_add": cls.torch_add,
            "torch_matmul": cls.torch_matmul,
            "torch_gelu": cls.torch_gelu,
            "torch_cuda_add": cls.torch_cuda_add,
            "cuda_vector_add": cls.cuda_vector_add,
            "cuda_matrix_multiply": cls.cuda_matrix_multiply,
            "triton_add": cls.triton_add,
            "triton_matmul": cls.triton_matmul,
            "compilation_error": cls.compilation_error,
            "runtime_error": cls.runtime_error,
            "validation_failure": cls.validation_failure,
        }
        
        kernel_fn = kernels.get(name)
        return kernel_fn() if kernel_fn else None
    
    @classmethod
    def get_kernels_by_type(cls, kernel_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all kernels of a specific type from test_data"""
        loader = get_loader()
        # Load all test cases and filter by kernel type
        all_test_cases = loader.get_all_test_cases()
        
        result = {}
        for name, test_case in all_test_cases:
            # Check custom kernel first
            kernel = loader.get_kernel_from_test_case(test_case, "custom_kernel")
            if kernel and kernel.get("kernel_type") == kernel_type:
                # Add metadata for torch kernels
                kernel = cls._add_torch_metadata(kernel)
                result[f"{name}_custom"] = kernel
            
            # Check ref kernel
            kernel = loader.get_kernel_from_test_case(test_case, "ref_kernel")
            if kernel and kernel.get("kernel_type") == kernel_type:
                # Add metadata for torch kernels
                kernel = cls._add_torch_metadata(kernel)
                result[f"{name}_ref"] = kernel
        
        return result