#!/usr/bin/env python3
"""
End-to-end test for custom tolerance feature
Tests that atol/rtol parameters flow through the API correctly
"""

import requests
import json

# Test configuration
SERVER_URL = "http://localhost:8000"

# Simple PyTorch reference kernel
ref_code = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2.0

    def get_inputs(self):
        return [torch.randn(100, device='cuda')]
"""

# Custom kernel with small difference
custom_code = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 2.0 + 0.005  # Small difference

    def get_inputs(self):
        return [torch.randn(100, device='cuda')]
"""

def test_with_default_tolerance():
    """Test comparison with default tolerance (1e-2) - should pass"""
    print("Test 1: Default tolerance (should pass)...")

    payload = {
        "ref_kernel": {
            "source_code": ref_code,
            "kernel_type": "torch"
        },
        "custom_kernel": {
            "source_code": custom_code,
            "kernel_type": "torch"
        },
        "num_trials": 10,
        "timeout": 60
    }

    response = requests.post(f"{SERVER_URL}/compare", json=payload)
    result = response.json()

    print(f"  Status: {response.status_code}")
    print(f"  Compiled: {result['kernel_exec_result']['compiled']}")
    print(f"  Correctness: {result['kernel_exec_result']['correctness']}")

    if not result['kernel_exec_result']['compiled']:
        print(f"  Compilation error: {result['kernel_exec_result'].get('compilation_error', 'N/A')}")

    if not result['kernel_exec_result']['correctness']:
        print(f"  Validation error: {result['kernel_exec_result'].get('validation_error', 'N/A')}")

    assert result['kernel_exec_result']['compiled'] == True, "Kernel should compile"
    assert result['kernel_exec_result']['correctness'] == True, "Should pass with default tolerance"
    print("  ✓ PASSED\n")

def test_with_strict_tolerance():
    """Test comparison with strict tolerance (1e-5) - should fail"""
    print("Test 2: Strict tolerance (should fail)...")

    payload = {
        "ref_kernel": {
            "source_code": ref_code,
            "kernel_type": "torch"
        },
        "custom_kernel": {
            "source_code": custom_code,
            "kernel_type": "torch"
        },
        "num_trials": 10,
        "timeout": 60,
        "atol": 1e-5,
        "rtol": 1e-5
    }

    response = requests.post(f"{SERVER_URL}/compare", json=payload)
    result = response.json()

    print(f"  Status: {response.status_code}")
    print(f"  Compiled: {result['kernel_exec_result']['compiled']}")
    print(f"  Correctness: {result['kernel_exec_result']['correctness']}")

    if result['kernel_exec_result']['correctness'] == False:
        print(f"  Error message: {result['kernel_exec_result'].get('validation_error', 'N/A')}")
        # Verify tolerance values appear in error message
        error_msg = result['kernel_exec_result'].get('validation_error', '')
        assert 'atol=1e-05' in error_msg or 'atol=0.00001' in error_msg, "Error should include atol value"
        assert 'rtol=1e-05' in error_msg or 'rtol=0.00001' in error_msg, "Error should include rtol value"

    assert result['kernel_exec_result']['correctness'] == False, "Should fail with strict tolerance"
    print("  ✓ PASSED\n")

def test_with_loose_tolerance():
    """Test comparison with loose tolerance (0.1) - should pass"""
    print("Test 3: Loose tolerance (should pass)...")

    payload = {
        "ref_kernel": {
            "source_code": ref_code,
            "kernel_type": "torch"
        },
        "custom_kernel": {
            "source_code": custom_code,
            "kernel_type": "torch"
        },
        "num_trials": 10,
        "timeout": 60,
        "atol": 0.1,
        "rtol": 0.1
    }

    response = requests.post(f"{SERVER_URL}/compare", json=payload)
    result = response.json()

    print(f"  Status: {response.status_code}")
    print(f"  Compiled: {result['kernel_exec_result']['compiled']}")
    print(f"  Correctness: {result['kernel_exec_result']['correctness']}")

    assert result['kernel_exec_result']['correctness'] == True, "Should pass with loose tolerance"
    print("  ✓ PASSED\n")

if __name__ == "__main__":
    print("=" * 60)
    print("End-to-End Tolerance Feature Test")
    print("=" * 60 + "\n")

    try:
        test_with_default_tolerance()
        test_with_strict_tolerance()
        test_with_loose_tolerance()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
