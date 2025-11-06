#!/usr/bin/env python3
"""
Test script to verify that func_ptr injection only targets specific C++ functions
and doesn't affect tensor methods like .size()

This test demonstrates the fix for the issue where ALL method calls were getting
func_ptr injected, including tensor methods.
"""

import ast
import sys
import os


def test_func_ptr_injection():
    """Test that func_ptr is only injected to the target function, not tensor methods"""
    
    # Example 1: RMS Norm with .size() calls
    source_code_1 = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        num_features = x.size(1)
        dim = x.size(2)
        return rms_norm.rms_norm_cuda(x, self.eps, batch_size, num_features, dim)
"""
    
    # Create a mock wrapper to access the injection method
    class MockWrapper:
        def _inject_func_ptr_parameter(self, source_code: str, cpp_function_name: str) -> str:
            """
            Adds 'func_ptr: int' parameter to forward method and passes it only to 
            calls matching the specified cpp_function_name.
            """
            
            class FuncPtrInjector(ast.NodeTransformer):
                def __init__(self, target_function_name):
                    self.in_forward = False
                    self.target_function_name = target_function_name
                    
                def visit_FunctionDef(self, node):
                    # Check if this is the forward method
                    if node.name == 'forward':
                        self.in_forward = True
                        
                        # Add func_ptr parameter to forward method signature
                        func_ptr_arg = ast.arg(
                            arg='func_ptr',
                            annotation=ast.Name(id='int', ctx=ast.Load())
                        )
                        
                        # Add it to the function's arguments
                        node.args.args.append(func_ptr_arg)
                        
                        # Process the body of forward method
                        node = self.generic_visit(node)
                        
                        self.in_forward = False
                    else:
                        node = self.generic_visit(node)
                        
                    return node
                
                def visit_Call(self, node):
                    # Only modify calls inside forward method
                    if self.in_forward:
                        # Check if this is an attribute call and matches our target
                        if isinstance(node.func, ast.Attribute):
                            # Check if the attribute name matches our target function
                            if node.func.attr == self.target_function_name:
                                # Add func_ptr as the last argument
                                node.args.append(ast.Name(id='func_ptr', ctx=ast.Load()))
                    
                    return self.generic_visit(node)
            
            # Parse the source code
            tree = ast.parse(source_code)
            
            # Apply the transformation
            transformer = FuncPtrInjector(cpp_function_name)
            modified_tree = transformer.visit(tree)
            
            # Fix missing locations in the AST
            ast.fix_missing_locations(modified_tree)
            
            # Convert back to source code
            if hasattr(ast, 'unparse'):
                return ast.unparse(modified_tree)
            else:
                # Fallback for Python < 3.9
                try:
                    import astor
                    return astor.to_source(modified_tree)
                except ImportError:
                    # For testing, just show that it would work
                    return "# AST modified but astor not available for conversion"
    
    mock_wrapper = MockWrapper()
    
    # Test 1: RMS Norm - should only inject func_ptr to rms_norm_cuda, not to .size()
    print("=" * 80)
    print("TEST 1: RMS Norm with .size() calls")
    print("=" * 80)
    modified_1 = mock_wrapper._inject_func_ptr_parameter(source_code_1, "rms_norm_cuda")
    print(modified_1)
    
    # Check that .size() calls don't have func_ptr
    assert "x.size(0, func_ptr)" not in modified_1, "ERROR: .size(0) should NOT have func_ptr!"
    assert "x.size(1, func_ptr)" not in modified_1, "ERROR: .size(1) should NOT have func_ptr!"
    assert "x.size(2, func_ptr)" not in modified_1, "ERROR: .size(2) should NOT have func_ptr!"
    
    # Check that rms_norm_cuda has func_ptr
    assert "rms_norm_cuda(x, self.eps, batch_size, num_features, dim, func_ptr)" in modified_1, \
        "ERROR: rms_norm_cuda should have func_ptr!"
    
    print("\n✅ Test 1 PASSED: Only rms_norm_cuda gets func_ptr, .size() methods are unchanged")
    
    # Example 2: Max Pooling
    source_code_2 = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.size(2)
        w = x.shape[3]  # Alternative way to get size
        return max_pooling1d.max_pooling1d_cuda(x, 2, 2, 0, 1, False, h)
"""
    
    print("\n" + "=" * 80)
    print("TEST 2: Max Pooling with .size() call")
    print("=" * 80)
    modified_2 = mock_wrapper._inject_func_ptr_parameter(source_code_2, "max_pooling1d_cuda")
    print(modified_2)
    
    # Check that .size() doesn't have func_ptr
    assert "x.size(2, func_ptr)" not in modified_2, "ERROR: .size(2) should NOT have func_ptr!"
    
    # Check that max_pooling1d_cuda has func_ptr
    assert "max_pooling1d_cuda(x, 2, 2, 0, 1, False, h, func_ptr)" in modified_2, \
        "ERROR: max_pooling1d_cuda should have func_ptr!"
    
    print("\n✅ Test 2 PASSED: Only max_pooling1d_cuda gets func_ptr")
    
    # Example 3: Multiple calls to same function
    source_code_3 = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        size_x = x.size(0)
        result1 = custom_kernel.process_cuda(x, size_x)
        
        size_y = y.size(0)
        result2 = custom_kernel.process_cuda(y, size_y)
        
        return result1 + result2
"""
    
    print("\n" + "=" * 80)
    print("TEST 3: Multiple calls to same CUDA function")
    print("=" * 80)
    modified_3 = mock_wrapper._inject_func_ptr_parameter(source_code_3, "process_cuda")
    print(modified_3)
    
    # Check that .size() calls don't have func_ptr
    assert "x.size(0, func_ptr)" not in modified_3, "ERROR: x.size(0) should NOT have func_ptr!"
    assert "y.size(0, func_ptr)" not in modified_3, "ERROR: y.size(0) should NOT have func_ptr!"
    
    # Check that both process_cuda calls have func_ptr
    assert "process_cuda(x, size_x, func_ptr)" in modified_3, \
        "ERROR: First process_cuda should have func_ptr!"
    assert "process_cuda(y, size_y, func_ptr)" in modified_3, \
        "ERROR: Second process_cuda should have func_ptr!"
    
    print("\n✅ Test 3 PASSED: Both process_cuda calls get func_ptr, .size() unchanged")
    
    # Example 4: Different method name (should not be modified)
    source_code_4 = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This function name doesn't match our target
        result = other_module.different_cuda(x, 10)
        # Only this one should get func_ptr
        result = custom.target_cuda(result, 20)
        return result
"""
    
    print("\n" + "=" * 80)
    print("TEST 4: Only target function gets modified")
    print("=" * 80)
    modified_4 = mock_wrapper._inject_func_ptr_parameter(source_code_4, "target_cuda")
    print(modified_4)
    
    # Check that different_cuda doesn't have func_ptr
    assert "different_cuda(x, 10, func_ptr)" not in modified_4, \
        "ERROR: different_cuda should NOT have func_ptr!"
    
    # Check that target_cuda has func_ptr
    assert "target_cuda(result, 20, func_ptr)" in modified_4, \
        "ERROR: target_cuda should have func_ptr!"
    
    print("\n✅ Test 4 PASSED: Only target_cuda gets func_ptr")
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 80)
    print("\nSummary:")
    print("✅ The fix correctly targets only the specified C++ function")
    print("✅ Tensor methods like .size() are not affected")
    print("✅ Multiple calls to the target function all get func_ptr")
    print("✅ Other function calls are not affected")


if __name__ == "__main__":
    test_func_ptr_injection()
