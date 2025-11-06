"""
End-to-end workflow tests for complete evaluation scenarios
"""

import pytest
import requests
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.kernels import KernelLibrary
from tests.fixtures.factories import RequestFactory, ResponseValidator, TestSuite, TestResult
from tests.fixtures.validators import TestTimer
from tests.fixtures.test_data_loader import get_loader


@pytest.mark.e2e
class TestCompleteWorkflows:
    """Tests for complete evaluation workflows"""
    
    def test_kernel_optimization_workflow(self, server_url):
        """Test complete kernel optimization workflow"""
        # Step 1: Baseline evaluation
        baseline_kernel = KernelLibrary.torch_add()
        
        baseline_request = RequestFactory.create_evaluate_request(
            kernel=baseline_kernel,
            num_trials=10
        )
        
        baseline_response = requests.post(f"{server_url}/evaluate", json=baseline_request)
        assert baseline_response.status_code == 200
        baseline_data = baseline_response.json()
        baseline_time = baseline_data["kernel_exec_result"]["runtime_stats"]["mean"]
        
        # Step 2: Compare with "optimized" version
        optimized_kernel = baseline_kernel.copy()
        optimized_kernel["kernel_name"] = "optimized_add"
        
        compare_request = RequestFactory.create_compare_request(
            kernel=baseline_kernel,
            custom_kernel=optimized_kernel,
            num_trials=10
        )
        
        compare_response = requests.post(f"{server_url}/compare", json=compare_request)
        assert compare_response.status_code == 200
        compare_data = compare_response.json()
        
        # Verify correctness
        assert compare_data["kernel_exec_result"]["correctness"]
        
        # Step 3: Detailed evaluation with metrics
        if compare_data["kernel_exec_result"]["correctness"]:
            detailed_request = RequestFactory.create_evaluate_request(
                kernel=optimized_kernel,
                num_trials=20
            )
            
            detailed_response = requests.post(f"{server_url}/evaluate", json=detailed_request)
            assert detailed_response.status_code == 200
    
    def test_multi_kernel_comparison_workflow(self, server_url):
        """Test comparing multiple kernel implementations"""
        implementations = [
            ("torch", KernelLibrary.torch_add()),
            pytest.param("triton", KernelLibrary.triton_add(), marks=pytest.mark.triton),
        ]
        
        results = {}
        baseline = None
        
        for impl_type, kernel in implementations:
            if isinstance(kernel, tuple):  # Skip parametrized tests that don't match
                continue
            
            # Evaluate each implementation
            request = RequestFactory.create_evaluate_request(
                kernel=kernel,
                num_trials=10
            )
            
            response = requests.post(f"{server_url}/evaluate", json=request)
            
            if response.status_code == 200:
                data = response.json()
                if data["kernel_exec_result"]["compiled"]:
                    perf = data["kernel_exec_result"]["runtime_stats"]["mean"]
                    results[impl_type] = perf
                    
                    if baseline is None:
                        baseline = kernel
        
        # Compare each against baseline
        for impl_type, kernel in implementations:
            if isinstance(kernel, tuple) or impl_type not in results:
                continue
            
            if baseline and kernel != baseline:
                compare_request = RequestFactory.create_compare_request(
                    kernel=baseline,
                    custom_kernel=kernel,
                    num_trials=5
                )
                
                response = requests.post(f"{server_url}/compare", json=compare_request)
                assert response.status_code == 200
    
    def test_progressive_optimization_workflow(self, server_url):
        """Test progressive kernel optimization with validation"""
        # Start with naive implementation
        v1_kernel = {
            "kernel_type": "torch",
            "kernel_name": "matmul_v1",
            "kernel_code": """
import torch

def kernel_fn(a, b):
    # Naive implementation
    return torch.matmul(a, b)
"""
        }
        
        # "Optimized" version (same for testing)
        v2_kernel = {
            "kernel_type": "torch",
            "kernel_name": "matmul_v2",
            "kernel_code": """
import torch

def kernel_fn(a, b):
    # "Optimized" implementation
    return torch.matmul(a, b)
"""
        }
        
        # Evaluate v1
        v1_request = RequestFactory.create_evaluate_request(
            kernel=v1_kernel,
            num_trials=5
        )
        
        v1_response = requests.post(f"{server_url}/evaluate", json=v1_request)
        assert v1_response.status_code == 200
        v1_data = v1_response.json()
        
        # Compare v1 vs v2
        compare_request = RequestFactory.create_compare_request(
            kernel=v1_kernel,
            custom_kernel=v2_kernel,
            num_trials=5
        )
        
        compare_response = requests.post(f"{server_url}/compare", json=compare_request)
        assert compare_response.status_code == 200
        compare_data = compare_response.json()
        
        # Verify correctness is maintained
        assert compare_data["kernel_exec_result"]["correctness"]
        
        # Check if performance improved (or at least didn't degrade much)
        if "ref_performance" in compare_data["results"] and "target_performance" in compare_data["results"]:
            ref_time = compare_data["kernel_exec_result"]["runtime_stats"]["mean"]
            target_time = compare_data["ref_runtime"]["mean"]
            
            # Should be within reasonable bounds
            assert target_time < ref_time * 1.5  # Not more than 50% slower


@pytest.mark.e2e
@pytest.mark.slow
class TestLongRunningWorkflows:
    """Tests for long-running evaluation scenarios"""
    
    def test_extended_benchmarking(self, server_url):
        """Test extended benchmarking with many trials"""
        kernel = KernelLibrary.torch_matmul()
        
        request = RequestFactory.create_evaluate_request(
            kernel=kernel,
            num_trials=100,  # Many trials for stable results
            timeout=120
        )
        
        with TestTimer() as timer:
            response = requests.post(f"{server_url}/evaluate", json=request, timeout=150)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that we got stable measurements
        perf = data["ref_runtime"]
        
        # Standard deviation should be relatively small compared to mean
        if perf["mean"] > 0:
            cv = perf["std"] / perf["mean"]  # Coefficient of variation
            assert cv < 0.5  # CV should be less than 50%
        
        # Should complete within timeout
        assert timer.duration() < 150
    
    def test_stress_test_workflow(self, server_url):
        """Test server under stress with multiple kernels"""
        kernels = [
            KernelLibrary.torch_add(),
            KernelLibrary.torch_matmul(),
            KernelLibrary.torch_gelu(),
        ]
        
        suite = TestSuite("Stress Test")
        
        for i, kernel in enumerate(kernels):
            request = RequestFactory.create_evaluate_request(
                kernel=kernel,
                num_trials=20
            )
            
            with TestTimer() as timer:
                response = requests.post(f"{server_url}/evaluate", json=request)
            
            if response.status_code == 200:
                data = response.json()
                passed = data["kernel_exec_result"]["compiled"]
            else:
                passed = False
            
            suite.add_test(TestResult(
                name=kernel["kernel_name"],
                passed=passed,
                duration=timer.duration()
            ))
        
        # Most tests should pass
        assert suite.success_rate() > 80


@pytest.mark.e2e
class TestRealWorldScenarios:
    """Tests simulating real-world usage patterns"""
    
    def test_model_layer_optimization(self, server_url):
        """Test optimizing individual model layers"""
        # Simulate optimizing different layers of a model
        layers = [
            {
                "name": "attention",
                "kernel": {
                    "kernel_type": "torch",
                    "kernel_name": "attention_layer",
                    "kernel_code": """
import torch
import torch.nn.functional as F

def kernel_fn(q, k, v):
    # Simplified attention
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = F.softmax(scores, dim=-1)
    return torch.matmul(scores, v)
"""
                }
            },
            {
                "name": "feedforward",
                "kernel": {
                    "kernel_type": "torch",
                    "kernel_name": "ffn_layer",
                    "kernel_code": """
import torch
import torch.nn.functional as F

def kernel_fn(x):
    # Simple FFN
    x = F.linear(x, torch.randn(1024, 768, device=x.device))
    x = F.gelu(x)
    x = F.linear(x, torch.randn(768, 1024, device=x.device))
    return x
"""
                }
            }
        ]
        
        for layer in layers:
            # Evaluate each layer
            request = RequestFactory.create_evaluate_request(
                kernel=layer["kernel"],
                num_trials=5,
                timeout=60
            )
            
            response = requests.post(f"{server_url}/evaluate", json=request)
            
            # Should handle various layer types
            assert response.status_code == 200
    
    def test_kernel_fusion_workflow(self, server_url):
        """Test kernel fusion optimization workflow"""
        # Separate kernels
        separate_kernel = {
            "kernel_type": "torch",
            "kernel_name": "separate_ops",
            "kernel_code": """
import torch

def kernel_fn(x):
    x = x * 2  # Scale
    x = x + 1  # Bias
    x = torch.relu(x)  # Activation
    return x
"""
        }
        
        # Fused kernel
        fused_kernel = {
            "kernel_type": "torch",
            "kernel_name": "fused_ops",
            "kernel_code": """
import torch

def kernel_fn(x):
    # Fused operations
    return torch.relu(x * 2 + 1)
"""
        }
        
        # Compare separate vs fused
        request = RequestFactory.create_compare_request(
            ref_kernel=separate_kernel,
            custom_kernel=fused_kernel,
            num_trials=10
        )
        
        response = requests.post(f"{server_url}/compare", json=request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should maintain correctness
        assert data["kernel_exec_result"]["correctness"]
        
        # Fused should generally be faster (or at least not slower)
        if "kernel_exec_result" in data:
            ref_time = data["kernel_exec_result"]["runtime_stats"]["mean"]
            target_time = data["ref_runtime"]["mean"]
            
            # Log the speedup
            if ref_time > 0:
                speedup = ref_time / target_time
                print(f"Fusion speedup: {speedup:.2f}x")


