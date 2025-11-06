"""
Test data loader for loading actual kernel test cases from JSON files
"""

import json
import os
import glob
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

class TestDataLoader:
    """Load test cases from JSON files in test_data directory"""
    
    def __init__(self, base_dir: str = None):
        """Initialize with base directory containing test_data"""
        if base_dir is None:
            # Get the project root (cuda_eval_server_v2)
            current_file = Path(__file__)
            self.base_dir = current_file.parent.parent.parent  # Up to cuda_eval_server_v2
        else:
            self.base_dir = Path(base_dir)
        
        self.test_data_dir = self.base_dir / "test_data"
        
        # Cache for loaded test cases
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a single JSON test case file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    
    def get_all_test_cases(self, kernel_type: str = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get all test cases, optionally filtered by kernel type
        
        Args:
            kernel_type: Filter by kernel type (cuda, triton, torch) or None for all
            
        Returns:
            List of (name, test_case) tuples
        """
        test_cases = []
        
        if kernel_type:
            # Load from specific subdirectory
            type_dir = self.test_data_dir / kernel_type
            if type_dir.exists():
                pattern = str(type_dir / "*.json")
            else:
                return test_cases
        else:
            # Load all JSON files
            pattern = str(self.test_data_dir / "**" / "*.json")
        
        json_files = glob.glob(pattern, recursive=True)
        json_files.sort()  # Ensure consistent order
        
        for file_path in json_files:
            # Skip some non-test files
            if "kernelbench_evaluated" in file_path or "kernelbench_generated" in file_path:
                continue
                
            test_case = self.load_json_file(file_path)
            if test_case:
                # Extract name from file path
                rel_path = Path(file_path).relative_to(self.test_data_dir)
                name = str(rel_path).replace('.json', '').replace('/', '_')
                test_cases.append((name, test_case))
                
                # Cache it
                self._cache[name] = test_case
        
        return test_cases
    
    def get_test_case(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific test case by name"""
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        # Try to load it
        # First try direct path
        for subdir in ['cuda', 'triton', '']:
            if subdir:
                file_path = self.test_data_dir / subdir / f"{name}.json"
            else:
                file_path = self.test_data_dir / f"{name}.json"
            
            if file_path.exists():
                test_case = self.load_json_file(str(file_path))
                if test_case:
                    self._cache[name] = test_case
                    return test_case
        
        # Try loading all and finding by name
        all_cases = self.get_all_test_cases()
        for case_name, test_case in all_cases:
            if case_name == name or case_name.endswith(name):
                self._cache[name] = test_case
                return test_case
        
        return None
    
    def get_kernel_from_test_case(self, test_case: Dict[str, Any], kernel_key: str = "custom_kernel") -> Optional[Dict[str, Any]]:
        """
        Extract a specific kernel from a test case
        
        Args:
            test_case: The loaded test case
            kernel_key: Which kernel to extract ("custom_kernel" or "ref_kernel")
            
        Returns:
            The kernel dict or None
        """
        return test_case.get(kernel_key)
    
    def get_cuda_test_cases(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all CUDA kernel test cases"""
        return self.get_all_test_cases("cuda")
    
    def get_triton_test_cases(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all Triton kernel test cases"""
        return self.get_all_test_cases("triton")
    
    def get_kernels_by_type(self, kernel_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all kernels of a specific type as a dict
        
        Returns:
            Dict mapping kernel name to kernel definition
        """
        test_cases = self.get_all_test_cases(kernel_type)
        kernels = {}
        
        for name, test_case in test_cases:
            # Try to get custom kernel first, then ref kernel
            kernel = self.get_kernel_from_test_case(test_case, "custom_kernel")
            if not kernel or kernel.get("kernel_type") != kernel_type:
                kernel = self.get_kernel_from_test_case(test_case, "ref_kernel")
            
            if kernel and kernel.get("kernel_type") == kernel_type:
                kernels[name] = kernel
        
        return kernels


# Global instance for convenience
_loader = None

def get_loader() -> TestDataLoader:
    """Get global test data loader instance"""
    global _loader
    if _loader is None:
        _loader = TestDataLoader()
    return _loader