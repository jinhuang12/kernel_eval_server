#!/usr/bin/env python3
"""
Integration test for TorchExecutableKernel function targeting via the eval server
Tests that the server correctly handles PyTorch kernels with function/method targeting
"""

import requests
import json
import sys
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_torch_function_targeting(base_url: str):
    """Test PyTorch function targeting through the server"""
    
    print("\n" + "="*60)
    print("TEST: PyTorch Function Targeting via Server")
    print("="*60)
    
    # Test 1: Standalone function
    print("\n1. Testing standalone function targeting...")
    
    source_code = """
import torch

def custom_add(x, y):
    return x + y * 2.0

def custom_multiply(x, y):
    return x * y

class Model(torch.nn.Module):
    def forward(self, x, y):
        # This should be ignored when function_name is specified
        return x - y
"""
    
    request_data = {
        "kernel": {
            "source_code": source_code,
            "kernel_type": "torch",
            "metadata": {
                "function_name": "custom_add"  # Target specific function
            },
            "io": {
                "args": [
                    {
                        "name": "x",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [512],
                            "dtype": "float32",
                            "init": {"kind": "ones"}
                        }
                    },
                    {
                        "name": "y",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [512],
                            "dtype": "float32",
                            "init": {"kind": "full", "fill_value": 3.0}
                        }
                    }
                ]
            }
        },
        "num_trials": 10,
        "timeout": 30
    }
    
    response = requests.post(f"{base_url}/evaluate", json=request_data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Standalone function executed successfully")
        print(f"  Job ID: {result.get('job_id', 'N/A')}")
        print(f"  Status: {result.get('status', 'N/A')}")
        
        if result.get("kernel_exec_result", {}).get("compilation_successful"):
            print("✓ Compilation successful")
        
        if result.get("kernel_exec_result", {}).get("validation_passed"):
            print("✓ Validation passed (function executed correctly)")
        
        runtime = result.get("kernel_exec_result", {}).get("avg_runtime_ms")
        if runtime:
            print(f"  Avg runtime: {runtime:.3f}ms")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text[:500]}")
        return False
    
    # Test 2: Class method targeting
    print("\n2. Testing class method targeting...")
    
    source_code = """
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 2.0
    
    def special_transform(self, x):
        return torch.relu(x - 0.5) * 3.0

class Model(nn.Module):
    def forward(self, x):
        return x + 1  # Should be ignored
"""
    
    request_data = {
        "kernel": {
            "source_code": source_code,
            "kernel_type": "torch",
            "metadata": {
                "class_name": "CustomModel",
                "method_name": "special_transform"  # Target specific method
            },
            "io": {
                "args": [
                    {
                        "name": "x",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [128, 256],
                            "dtype": "float32",
                            "init": {"kind": "randn"}
                        }
                    }
                ]
            }
        },
        "num_trials": 10,
        "timeout": 30
    }
    
    response = requests.post(f"{base_url}/evaluate", json=request_data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Class method executed successfully")
        print(f"  Job ID: {result.get('job_id', 'N/A')}")
        
        if result.get("kernel_exec_result", {}).get("compilation_successful"):
            print("✓ Compilation successful")
        
        runtime = result.get("kernel_exec_result", {}).get("avg_runtime_ms")
        if runtime:
            print(f"  Avg runtime: {runtime:.3f}ms")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        return False
    
    # Test 3: Default behavior (no metadata)
    print("\n3. Testing default behavior without metadata...")
    
    source_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sigmoid(x)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(64, 128)]
"""
    
    request_data = {
        "kernel": {
            "source_code": source_code,
            "kernel_type": "torch",
            # No metadata - should use traditional KernelBench pattern
        },
        "num_trials": 10,
        "timeout": 30
    }
    
    response = requests.post(f"{base_url}/evaluate", json=request_data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Default behavior (KernelBench pattern) worked")
        print(f"  Job ID: {result.get('job_id', 'N/A')}")
        
        if result.get("kernel_exec_result", {}).get("compilation_successful"):
            print("✓ Compilation successful")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        return False
    
    # Test 4: Complex shit
    print("\n4. Testing complex shit...")
    source_code = """
import torch
import torch.nn.functional as F

# Reference MoE forward (SwiGLU), vectorized over top-2 routing.
# Shapes:
#   X: [T, H], TOPK_IDS: [T, K], TOPK_W: [T, K]
#   W1: [E, H, Fdim], W3: [E, H, Fdim], W2: [E, Fdim, H]
# Returns:
#   Y: [T, H]

def ref_kernel(X, TOPK_IDS, TOPK_W, W1, W3, W2, Y=None):
    assert X.dim() == 2, "X must be [T, H]"
    T, H = X.shape
    E, H1, Fdim = W1.shape
    assert H1 == H, "W1 last-2 dims must match hidden size"
    assert W3.shape == (E, H, Fdim), "W3 must match [E, H, Fdim]"
    assert W2.shape == (E, Fdim, H), "W2 must match [E, Fdim, H]"
    assert TOPK_IDS.shape == (T, TOPK_W.shape[1]), "top-k ids/weights mismatch"

    K = TOPK_IDS.shape[1]
    # normalize router weights along K to avoid scale drift
    w = TOPK_W.to(X.dtype) / (TOPK_W.sum(dim=-1, keepdim=True) + 1e-6)  # [T, K]

    # Gather expert weights per token/choice
    # W1_e: [T, K, H, Fdim], W3_e: [T, K, H, Fdim], W2_e: [T, K, Fdim, H]
    W1_e = W1[TOPK_IDS]  # advanced indexing requires int64/long
    W3_e = W3[TOPK_IDS]
    W2_e = W2[TOPK_IDS]

    # Expand inputs to [T, K, H]
    Xk = X.unsqueeze(1).expand(T, K, H)

    # Project: [T, K, Fdim]
    A = torch.einsum('tkh,tkhf->tkf', Xk, W1_e)
    B = torch.einsum('tkh,tkhf->tkf', Xk, W3_e)

    # SwiGLU gate and down-projection: [T, K, H]
    G = F.silu(A) * B
    Yk = torch.einsum('tkf,tkfh->tkh', G, W2_e)

    # Weighted sum over K: [T, H]
    Y_out = (Yk * w.unsqueeze(-1)).sum(dim=1)
    if Y is None:
        return Y_out
    else:
        Y.copy_(Y_out)
        return Y
"""

    request_data = {
        "kernel": {
            "source_code": source_code,
            "kernel_type": "torch",
            "metadata": {
                "function_name": "ref_kernel"
            },
            "io": {
                "args": [
                    {
                        "name": "X",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [2048, 1024],
                            "dtype": "float16",
                            "init": { "kind": "randn", "seed": 7 }
                        }
                    },
                    {
                        "name": "TOPK_IDS",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [2048, 2],
                            "dtype": "int64",
                            "init": { "kind": "ones" }
                        }
                    },
                    {
                        "name": "TOPK_W",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [2048, 2],
                            "dtype": "float16",
                            "init": { "kind": "uniform", "low": 0.0, "high": 1.0, "seed": 17 }
                        }
                    },
                    {
                        "name": "W1",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [16, 1024, 4096],
                            "dtype": "float16",
                            "init": { "kind": "randn", "seed": 23 }
                        }
                    },
                    {
                        "name": "W3",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [16, 1024, 4096],
                            "dtype": "float16",
                            "init": { "kind": "randn", "seed": 29 }
                        }
                    },
                    {
                    "name": "W2",
                        "type": "tensor",
                        "role": "input",
                        "tensor_spec": {
                            "shape": [16, 4096, 1024],
                            "dtype": "float16",
                            "init": { "kind": "randn", "seed": 31 }
                        }
                    },
                    {
                        "name": "Y",
                        "type": "tensor",
                        "role": "output",
                        "tensor_spec": { "shape": [2048, 1024], "dtype": "float16" }
                    }
                ],
            }
        },
        "num_trials": 10,
        "timeout": 30
    }
    response = requests.post(f"{base_url}/evaluate", json=request_data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Complex behavior worked")
        print(f"  Job ID: {result.get('job_id', 'N/A')}")
        
        if result.get("kernel_exec_result", {}).get("compilation_successful"):
            print("✓ Compilation successful")
    else:
        print(f"✗ Request failed with status {response.status_code}")
        return False
    

    print("\n" + "="*60)
    print("✓ All PyTorch function targeting tests passed!")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test PyTorch function targeting via eval server")
    parser.add_argument("base_url", help="Base URL of the eval server (e.g., http://localhost:8000)")
    args = parser.parse_args()
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    try:
        success = test_torch_function_targeting(args.base_url)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()