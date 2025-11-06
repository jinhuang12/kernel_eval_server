#!/usr/bin/env python3
"""
Test Suite for Single Kernel Evaluation Endpoint
Tests the /evaluate endpoint for various kernel types
"""

import requests
import json
import time
import logging
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestCase:
    """Container for a test case"""
    def __init__(self, name: str, kernel_code: str, kernel_type: str, description: str = ""):
        self.name = name
        self.kernel_code = kernel_code
        self.kernel_type = kernel_type
        self.description = description


class SingleKernelEvaluationTester:
    """Test harness for single kernel evaluation endpoint"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.test_results = []
    
    def test_kernel(self, test_case: TestCase, num_trials: int = 10) -> Dict[str, Any]:
        """
        Test a single kernel evaluation
        
        Args:
            test_case: Test case to run
            num_trials: Number of profiling trials
            
        Returns:
            Test result dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test_case.name}")
        logger.info(f"Kernel Type: {test_case.kernel_type}")
        logger.info(f"Description: {test_case.description}")
        logger.info(f"{'='*60}")
        
        # Prepare request
        request_data = {
            "kernel": {
                "source_code": test_case.kernel_code,
                "kernel_type": test_case.kernel_type
            },
            "num_trials": num_trials,
            "timeout": 120
        }
        
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/evaluate",
                json=request_data,
                timeout=130
            )
            elapsed_time = time.time() - start_time
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                
                # Validate response structure
                assert "job_id" in result, "Missing job_id in response"
                assert "kernel_exec_result" in result, "Missing kernel_exec_result in response"
                assert "status" in result, "Missing status in response"
                
                kernel_result = result['kernel_exec_result']
                
                # Log results
                logger.info(f"✓ Test passed: {test_case.name}")
                logger.info(f"  Job ID: {result['job_id']}")
                logger.info(f"  Compiled: {kernel_result['compiled']}")
                logger.info(f"  Correctness: {kernel_result.get('correctness', 'N/A')}")
                
                if kernel_result['compiled']:
                    runtime_stats = kernel_result.get('runtime_stats', {})
                    if runtime_stats:
                        logger.info(f"  Runtime (mean): {runtime_stats.get('mean', 'N/A')} ms")
                        logger.info(f"  Runtime (min): {runtime_stats.get('min', 'N/A')} ms")
                        logger.info(f"  Runtime (max): {runtime_stats.get('max', 'N/A')} ms")
                    else:
                        logger.info(f"  Runtime: {kernel_result.get('runtime', 'N/A')} ms")
                    
                    metadata = kernel_result.get('metadata', {})
                    if metadata.get('device_metrics'):
                        logger.info(f"  Device metrics collected: Yes")
                else:
                    logger.info(f"  Compilation error: {kernel_result.get('compilation_error', 'Unknown')}")
                
                if not kernel_result.get('correctness', True):
                    logger.info(f"  Validation error: {kernel_result.get('validation_error', 'Unknown')}")
                
                logger.info(f"  Response time: {elapsed_time:.2f}s")
                
                return {
                    "test_name": test_case.name,
                    "success": True,
                    "compiled": kernel_result['compiled'],
                    "runtime_stats": kernel_result.get('runtime_stats', {}),
                    "response_time": elapsed_time,
                    "error": kernel_result.get('compilation_error') or kernel_result.get('validation_error')
                }
                
            else:
                error_msg = f"Request failed with status {response.status_code}"
                if response.text:
                    error_msg += f": {response.text}"
                logger.error(f"✗ Test failed: {test_case.name}")
                logger.error(f"  Error: {error_msg}")
                
                return {
                    "test_name": test_case.name,
                    "success": False,
                    "error": error_msg,
                    "response_time": elapsed_time
                }
                
        except Exception as e:
            logger.error(f"✗ Test failed: {test_case.name}")
            logger.error(f"  Exception: {str(e)}")
            
            return {
                "test_name": test_case.name,
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def run_all_tests(self):
        """Run all test cases"""
        test_cases = self._get_all_test_cases()
        
        logger.info(f"\nRunning {len(test_cases)} test cases...")
        
        for test_case in test_cases:
            result = self.test_kernel(test_case)
            self.test_results.append(result)
        
        # Print summary
        self._print_summary()
    
    def _get_all_test_cases(self) -> list:
        """Get all test cases"""
        test_cases = []
        
        # TORCH kernel test
        test_cases.append(TestCase(
            name="torch_simple_add",
            kernel_type="torch",
            description="Simple PyTorch addition model",
            kernel_code='''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return x + y

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.randn(1024, 1024, device='cuda')
    return [x, y]
'''
        ))
        
        # TORCH_CUDA kernel test
        test_cases.append(TestCase(
            name="torch_cuda_vector_add",
            kernel_type="torch_cuda",
            description="PyTorch with embedded CUDA kernel",
            kernel_code='''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    return c;
}
"""

cpp_source = """
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
"""

module = load_inline(
    name='vector_add_cuda',
    cpp_sources=cpp_source,
    cuda_sources=[cuda_source],
    functions=['vector_add'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return module.vector_add(x, y)

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
    return [x, y]
'''
        ))
        
        # TRITON kernel test
        test_cases.append(TestCase(
            name="triton_vector_add",
            kernel_type="triton",
            description="Triton kernel for vector addition",
            kernel_code='''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def run_add():
    size = 1024 * 1024
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
    return output
'''
        ))
        
        # Compilation failure test
        test_cases.append(TestCase(
            name="compilation_failure",
            kernel_type="torch",
            description="Test handling of compilation failure",
            kernel_code='''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Intentional syntax error
        return x + undefined_variable

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(1024, device='cuda')]
'''
        ))
        
        return test_cases
    
    def _print_summary(self):
        """Print test summary"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['success'])
        compiled = sum(1 for r in self.test_results if r.get('compiled', False))
        
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        logger.info(f"Successfully compiled: {compiled}/{total}")
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r['success']]
        if failed_tests:
            logger.info("\nFailed tests:")
            for test in failed_tests:
                logger.info(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")
        
        # Show compilation failures
        compile_failed = [r for r in self.test_results if r['success'] and not r.get('compiled', False)]
        if compile_failed:
            logger.info("\nCompilation failures:")
            for test in compile_failed:
                logger.info(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")
        
        logger.info(f"{'='*60}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test single kernel evaluation endpoint')
    parser.add_argument('--server', default='http://localhost:8000',
                       help='Server URL (default: http://localhost:8000)')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of profiling trials (default: 10)')
    
    args = parser.parse_args()
    
    # Check server health
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"Server health check failed: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Cannot connect to server at {args.server}: {e}")
        sys.exit(1)
    
    logger.info(f"Server is healthy at {args.server}")
    
    # Run tests
    tester = SingleKernelEvaluationTester(args.server)
    tester.run_all_tests()
    
    # Exit with appropriate code
    if all(r['success'] for r in tester.test_results):
        logger.info("\n✓ All tests passed!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()