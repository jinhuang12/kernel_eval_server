#!/usr/bin/env python3
"""
Comprehensive test suite for IOContract and related data models
Tests TensorData compression/decompression, TensorSpec, ArgSpec, LaunchConfig
"""

import asyncio
import base64
import json
import logging
import numpy as np
import torch
import sys
import os
import zlib
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from shared.models import (
    TensorData, TensorSpec, TensorInit, ArgSpec, 
    IOContract, LaunchConfig, LaunchDim
)
from shared.utils import (
    encode_tensor_to_data, decode_tensor_from_data, materialize_tensor
)


# ============================================================================
# Test 1: TensorData Compression/Decompression
# ============================================================================

async def test_tensor_data_compression():
    """Test TensorData compression and decompression with various configurations"""
    print("\n" + "="*80)
    print("🧪 TEST 1: TensorData Compression/Decompression")
    print("="*80)
    
    test_cases = []
    
    # Different shapes
    test_cases.extend([
        ("1D Vector", torch.randn(1024)),
        ("2D Matrix", torch.randn(64, 128)),
        ("3D Tensor", torch.randn(8, 32, 64)),
        ("4D Tensor", torch.randn(4, 8, 16, 32)),
        ("Scalar", torch.tensor(3.14159)),
        ("Single Element", torch.randn(1)),
        ("Empty Tensor", torch.empty(0)),
        ("Large Tensor", torch.randn(512, 512)),
    ])
    
    # Different dtypes
    dtypes_to_test = [
        (torch.float64, "float64"),
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.int64, "int64"),
        (torch.int32, "int32"),
        (torch.int16, "int16"),
        (torch.int8, "int8"),
        (torch.uint8, "uint8"),
        (torch.bool, "bool"),
    ]
    
    for dtype, name in dtypes_to_test:
        if dtype == torch.bool:
            test_cases.append((f"Bool tensor ({name})", torch.randint(0, 2, (32, 32), dtype=dtype)))
        elif dtype in [torch.int8, torch.uint8]:
            test_cases.append((f"Int8 tensor ({name})", torch.randint(-128 if dtype == torch.int8 else 0, 
                                                                       127 if dtype == torch.int8 else 255, 
                                                                       (32, 32), dtype=dtype)))
        elif dtype.is_floating_point:
            test_cases.append((f"Float tensor ({name})", torch.randn(32, 32, dtype=dtype)))
        else:
            test_cases.append((f"Int tensor ({name})", torch.randint(-1000, 1000, (32, 32), dtype=dtype)))
    
    # Test with and without compression
    compression_modes = ["none", "zlib"]
    
    passed = 0
    failed = 0
    
    for name, tensor in test_cases:
        for compress in compression_modes:
            test_name = f"{name} with {compress} compression"
            
            try:
                # Encode tensor to TensorData
                tensor_data = encode_tensor_to_data(tensor, compress=compress)
                
                # Verify TensorData fields
                assert tensor_data.shape == list(tensor.shape), f"Shape mismatch: {tensor_data.shape} != {list(tensor.shape)}"
                assert tensor_data.compress == compress, f"Compression mode mismatch: {tensor_data.compress} != {compress}"
                
                # Verify data is base64 encoded
                try:
                    decoded_b64 = base64.b64decode(tensor_data.data_b64)
                    if compress == "zlib":
                        # Should be compressed
                        raw_data = zlib.decompress(decoded_b64)
                    else:
                        raw_data = decoded_b64
                except Exception as e:
                    raise AssertionError(f"Failed to decode base64 data: {e}")
                
                # Decode back to tensor
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                decoded_tensor = decode_tensor_from_data(tensor_data, device=device)
                
                # Move original tensor to same device for comparison
                original_on_device = tensor.to(device)
                
                # Verify tensor matches
                if tensor.numel() > 0:  # Skip comparison for empty tensors
                    if tensor.dtype == torch.bool:
                        assert torch.equal(decoded_tensor, original_on_device), f"Bool tensor mismatch for {test_name}"
                    elif tensor.dtype.is_floating_point:
                        # For floating point, use allclose due to potential precision differences
                        if tensor.dtype == torch.float16:
                            # float16 has lower precision
                            assert torch.allclose(decoded_tensor, original_on_device, atol=1e-3, rtol=1e-3), \
                                f"Float tensor mismatch for {test_name}"
                        else:
                            assert torch.allclose(decoded_tensor, original_on_device, atol=1e-6), \
                                f"Float tensor mismatch for {test_name}"
                    else:
                        assert torch.equal(decoded_tensor, original_on_device), f"Int tensor mismatch for {test_name}"
                
                # Verify shape and dtype preserved
                assert decoded_tensor.shape == tensor.shape, f"Shape not preserved for {test_name}"
                assert decoded_tensor.dtype == tensor.dtype, f"Dtype not preserved for {test_name}"
                
                print(f"✅ {test_name}: PASSED")
                passed += 1
                
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
                failed += 1
    
    # Test CUDA tensors if available
    if torch.cuda.is_available():
        print("\n📝 Testing CUDA tensors...")
        cuda_tensor = torch.randn(128, 256, device='cuda')
        
        try:
            # Encode CUDA tensor (should automatically move to CPU)
            tensor_data = encode_tensor_to_data(cuda_tensor, compress="zlib")
            
            # Decode back to CUDA
            decoded = decode_tensor_from_data(tensor_data, device='cuda')
            
            assert decoded.device.type == 'cuda', "Tensor not on CUDA device"
            assert torch.allclose(decoded, cuda_tensor), "CUDA tensor values don't match"
            
            print("✅ CUDA tensor compression/decompression: PASSED")
            passed += 1
            
        except Exception as e:
            print(f"❌ CUDA tensor test: FAILED - {e}")
            failed += 1
    
    print(f"\n📊 Compression Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 2: TensorSpec with TensorInit
# ============================================================================

async def test_tensor_init():
    """Test TensorSpec with various TensorInit configurations"""
    print("\n" + "="*80)
    print("🧪 TEST 2: TensorSpec with TensorInit")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = [
        # randn with seed
        ("randn with seed", 
         TensorSpec(shape=[100], dtype="float32", 
                   init=TensorInit(kind="randn", seed=42))),
        
        # randn with mean and std
        ("randn with mean/std",
         TensorSpec(shape=[50, 50], dtype="float32",
                   init=TensorInit(kind="randn", seed=42, mean=5.0, std=2.0))),
        
        # zeros
        ("zeros",
         TensorSpec(shape=[32, 64], dtype="float32",
                   init=TensorInit(kind="zeros"))),
        
        # ones
        ("ones",
         TensorSpec(shape=[128], dtype="int32",
                   init=TensorInit(kind="ones"))),
        
        # uniform
        ("uniform",
         TensorSpec(shape=[10, 10], dtype="float32",
                   init=TensorInit(kind="uniform", low=-1.0, high=1.0, seed=123))),
        
        # full
        ("full with value",
         TensorSpec(shape=[8, 8, 8], dtype="float32",
                   init=TensorInit(kind="full", fill_value=3.14))),
        
        # arange
        ("arange",
         TensorSpec(shape=[20], dtype="float32",
                   init=TensorInit(kind="arange", start=0.0, step=0.5))),
    ]
    
    passed = 0
    failed = 0
    
    for name, spec in test_cases:
        try:
            # Materialize tensor from spec
            tensor = materialize_tensor(spec, default_device=device)
            
            # Verify shape
            assert list(tensor.shape) == spec.shape, f"Shape mismatch for {name}"
            
            # Verify dtype
            expected_dtype = {
                "float64": torch.float64,
                "float32": torch.float32,
                "float16": torch.float16,
                "int64": torch.int64,
                "int32": torch.int32,
                "int16": torch.int16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "bool": torch.bool,
            }[spec.dtype]
            assert tensor.dtype == expected_dtype, f"Dtype mismatch for {name}"
            
            # Verify specific properties based on init kind
            if spec.init.kind == "zeros":
                assert torch.all(tensor == 0), f"Not all zeros for {name}"
            elif spec.init.kind == "ones":
                assert torch.all(tensor == 1), f"Not all ones for {name}"
            elif spec.init.kind == "full":
                assert torch.allclose(tensor, torch.full(tensor.shape, spec.init.fill_value, dtype=tensor.dtype, device=device)), \
                    f"Not filled with correct value for {name}"
            elif spec.init.kind == "randn" and spec.init.mean is not None:
                # Check mean is approximately correct (with large enough sample)
                if tensor.numel() >= 100:
                    actual_mean = tensor.mean().item()
                    assert abs(actual_mean - spec.init.mean) < 1.0, \
                        f"Mean too far from expected for {name}: {actual_mean} vs {spec.init.mean}"
            elif spec.init.kind == "uniform":
                assert torch.all(tensor >= spec.init.low) and torch.all(tensor <= spec.init.high), \
                    f"Values outside uniform range for {name}"
            elif spec.init.kind == "arange":
                expected = torch.arange(spec.init.start, 
                                      spec.init.start + spec.init.step * spec.shape[0],
                                      spec.init.step, dtype=tensor.dtype, device=device)[:spec.shape[0]]
                # Reshape if needed
                if len(spec.shape) > 1:
                    expected = expected.reshape([1] * (len(spec.shape) - 1) + [spec.shape[-1]])
                    expected = expected.expand(spec.shape).clone()
                assert torch.allclose(tensor, expected), f"Arange values incorrect for {name}"
            
            # Test determinism with seed
            if spec.init.seed is not None:
                tensor2 = materialize_tensor(spec, default_device=device)
                assert torch.equal(tensor, tensor2), f"Not deterministic with seed for {name}"
            
            print(f"✅ {name}: PASSED (shape={tensor.shape}, dtype={tensor.dtype})")
            passed += 1
            
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
    
    print(f"\n📊 TensorInit Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 3: TensorSpec with TensorData
# ============================================================================

async def test_tensor_spec_with_data():
    """Test TensorSpec with literal TensorData"""
    print("\n" + "="*80)
    print("🧪 TEST 3: TensorSpec with TensorData")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create some test tensors
    test_tensors = [
        ("Small float tensor", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("2D int tensor", torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)),
        ("Bool tensor", torch.tensor([True, False, True, False], dtype=torch.bool)),
        ("Large random tensor", torch.randn(100, 100)),
    ]
    
    passed = 0
    failed = 0
    
    for name, original_tensor in test_tensors:
        for compress in ["none", "zlib"]:
            test_name = f"{name} with {compress} compression"
            
            try:
                # Create TensorData from tensor
                tensor_data = encode_tensor_to_data(original_tensor, compress=compress)
                
                # Create TensorSpec with literal data
                spec = TensorSpec(
                    shape=list(original_tensor.shape),
                    dtype={
                        torch.float64: "float64",
                        torch.float32: "float32",
                        torch.float16: "float16",
                        torch.int64: "int64",
                        torch.int32: "int32",
                        torch.int16: "int16",
                        torch.int8: "int8",
                        torch.uint8: "uint8",
                        torch.bool: "bool",
                    }[original_tensor.dtype],
                    data=tensor_data  # Literal data instead of init
                )
                
                # Materialize tensor from spec
                materialized = materialize_tensor(spec, default_device=device)
                
                # Compare with original
                original_on_device = original_tensor.to(device)
                
                if original_tensor.dtype == torch.bool:
                    assert torch.equal(materialized, original_on_device), f"Bool tensor mismatch for {test_name}"
                elif original_tensor.dtype.is_floating_point:
                    assert torch.allclose(materialized, original_on_device), f"Float tensor mismatch for {test_name}"
                else:
                    assert torch.equal(materialized, original_on_device), f"Int tensor mismatch for {test_name}"
                
                print(f"✅ {test_name}: PASSED")
                passed += 1
                
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
                failed += 1
    
    # Test mutual exclusivity (can't have both init and data)
    print("\n📝 Testing mutual exclusivity of init and data...")
    try:
        invalid_spec = TensorSpec(
            shape=[10],
            dtype="float32",
            init=TensorInit(kind="randn"),
            data=TensorData(data_b64="dummy", dtype="float32", shape=[10])
        )
        # This should ideally raise an error or ignore one
        tensor = materialize_tensor(invalid_spec)
        print("⚠️ Mutual exclusivity not enforced (tensor created anyway)")
        
    except Exception as e:
        print(f"✅ Mutual exclusivity enforced: {e}")
        passed += 1
    
    print(f"\n📊 TensorData Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 4: ArgSpec Tests
# ============================================================================

async def test_arg_spec():
    """Test ArgSpec configurations for different argument types"""
    print("\n" + "="*80)
    print("🧪 TEST 4: ArgSpec Tests")
    print("="*80)
    
    test_cases = []
    
    # Tensor arguments with different roles
    test_cases.extend([
        ("Input tensor", ArgSpec(
            name="x",
            type="tensor",
            role="input",
            tensor_spec=TensorSpec(shape=[32, 64], dtype="float32", init=TensorInit(kind="randn"))
        )),
        ("Output tensor", ArgSpec(
            name="output",
            type="tensor",
            role="output",
            tensor_spec=TensorSpec(shape=[32, 64], dtype="float32")
        )),
        ("Inout tensor", ArgSpec(
            name="buffer",
            type="tensor",
            role="inout",
            tensor_spec=TensorSpec(shape=[128], dtype="int32", init=TensorInit(kind="zeros"))
        )),
    ])
    
    # Scalar arguments
    test_cases.extend([
        ("Int scalar", ArgSpec(name="batch_size", type="int", value=32)),
        ("Float scalar", ArgSpec(name="learning_rate", type="float", value=0.001)),
        ("Bool scalar", ArgSpec(name="training", type="bool", value=True)),
        ("String scalar", ArgSpec(name="mode", type="str", value="inference")),
    ])
    
    # Meta parameters (Triton constexpr)
    test_cases.extend([
        ("Meta parameter", ArgSpec(
            name="BLOCK_SIZE",
            type="int",
            value=256,
            is_meta=True
        )),
    ])
    
    passed = 0
    failed = 0
    
    for name, arg_spec in test_cases:
        try:
            # Test to_dict serialization
            arg_dict = arg_spec.to_dict()
            
            # Verify required fields
            assert "name" in arg_dict, f"Missing 'name' in dict for {name}"
            assert "type" in arg_dict, f"Missing 'type' in dict for {name}"
            assert arg_dict["name"] == arg_spec.name, f"Name mismatch for {name}"
            assert arg_dict["type"] == arg_spec.type, f"Type mismatch for {name}"
            
            # Verify optional fields
            if arg_spec.value is not None:
                assert "value" in arg_dict, f"Missing 'value' in dict for {name}"
                assert arg_dict["value"] == arg_spec.value, f"Value mismatch for {name}"
            
            if arg_spec.tensor_spec is not None:
                assert "tensor_spec" in arg_dict, f"Missing 'tensor_spec' in dict for {name}"
                
            if arg_spec.role != "input":  # input is default
                assert "role" in arg_dict, f"Missing 'role' in dict for {name}"
                assert arg_dict["role"] == arg_spec.role, f"Role mismatch for {name}"
            
            # Verify tensor specs can be materialized
            if arg_spec.type == "tensor" and arg_spec.tensor_spec and arg_spec.tensor_spec.init:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                tensor = materialize_tensor(arg_spec.tensor_spec, default_device=device)
                assert tensor is not None, f"Failed to materialize tensor for {name}"
            
            print(f"✅ {name}: PASSED")
            passed += 1
            
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
    
    print(f"\n📊 ArgSpec Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 5: LaunchConfig Tests
# ============================================================================

async def test_launch_config():
    """Test LaunchConfig with various grid and block configurations"""
    print("\n" + "="*80)
    print("🧪 TEST 5: LaunchConfig Tests")
    print("="*80)
    
    test_cases = [
        ("1D Grid", LaunchConfig(grid=LaunchDim(x=256))),
        ("2D Grid", LaunchConfig(grid=LaunchDim(x=16, y=32))),
        ("3D Grid", LaunchConfig(grid=LaunchDim(x=8, y=8, z=4))),
        ("Grid with Block", LaunchConfig(
            grid=LaunchDim(x=32, y=16),
            block=LaunchDim(x=256)
        )),
        ("Full CUDA Config", LaunchConfig(
            grid=LaunchDim(x=64, y=64),
            block=LaunchDim(x=16, y=16)
        )),
        ("Triton Config", LaunchConfig(
            grid=LaunchDim(x=128),
            num_warps=4,
            num_stages=3
        )),
    ]
    
    passed = 0
    failed = 0
    
    for name, config in test_cases:
        try:
            # Verify grid dimensions
            if config.grid:
                assert config.grid.x >= 1, f"Invalid grid.x for {name}"
                assert config.grid.y >= 1, f"Invalid grid.y for {name}"
                assert config.grid.z >= 1, f"Invalid grid.z for {name}"
            
            # Verify block dimensions if present
            if config.block:
                assert config.block.x >= 1, f"Invalid block.x for {name}"
                assert config.block.y >= 1, f"Invalid block.y for {name}"
                assert config.block.z >= 1, f"Invalid block.z for {name}"
            
            if config.num_warps is not None:
                assert config.num_warps > 0, f"Invalid num_warps for {name}"
            
            if config.num_stages is not None:
                assert config.num_stages > 0, f"Invalid num_stages for {name}"
            
            print(f"✅ {name}: PASSED")
            passed += 1
            
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
    
    print(f"\n📊 LaunchConfig Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 6: IOContract Integration
# ============================================================================

async def test_io_contract_integration():
    """Test complete IOContract with all components"""
    print("\n" + "="*80)
    print("🧪 TEST 6: IOContract Integration")
    print("="*80)
    
    # Create a comprehensive IOContract
    io_contract = IOContract(
        args=[
            # Input tensors with TensorInit
            ArgSpec(
                name="input_a",
                type="tensor",
                role="input",
                tensor_spec=TensorSpec(
                    shape=[64, 128],
                    dtype="float32",
                    init=TensorInit(kind="randn", seed=42)
                )
            ),
            # Input tensor with literal TensorData
            ArgSpec(
                name="input_b",
                type="tensor",
                role="input",
                tensor_spec=TensorSpec(
                    shape=[128, 256],
                    dtype="float32",
                    data=encode_tensor_to_data(torch.ones(128, 256), compress="zlib")
                )
            ),
            # Output tensor
            ArgSpec(
                name="output",
                type="tensor",
                role="output",
                tensor_spec=TensorSpec(shape=[64, 256], dtype="float32")
            ),
            # Scalar arguments
            ArgSpec(name="alpha", type="float", value=1.5),
            ArgSpec(name="beta", type="float", value=0.5),
            ArgSpec(name="transpose", type="bool", value=False),
            # Meta parameters
            ArgSpec(name="BLOCK_SIZE_M", type="int", value=32, is_meta=True),
            ArgSpec(name="BLOCK_SIZE_N", type="int", value=64, is_meta=True),
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=2, y=4),
            num_warps=4
        )
    )
    
    passed = 0
    failed = 0
    
    # Test 1: Verify structure
    try:
        assert len(io_contract.args) == 8, f"Expected 8 args, got {len(io_contract.args)}"
        assert io_contract.launch is not None, "Missing launch config"
        
        print("✅ IOContract structure: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ IOContract structure: FAILED - {e}")
        failed += 1
    
    # Test 2: Materialize all tensors
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        materialized_tensors = []
        
        for arg in io_contract.args:
            if arg.type == "tensor" and arg.tensor_spec:
                if arg.role != "output":  # Don't initialize outputs
                    tensor = materialize_tensor(arg.tensor_spec, default_device=device)
                    materialized_tensors.append((arg.name, tensor))
        
        # Verify we got the expected tensors
        assert len(materialized_tensors) == 2, f"Expected 2 input tensors, got {len(materialized_tensors)}"
        
        # Verify shapes
        for name, tensor in materialized_tensors:
            if name == "input_a":
                assert tensor.shape == torch.Size([64, 128]), f"Wrong shape for {name}"
            elif name == "input_b":
                assert tensor.shape == torch.Size([128, 256]), f"Wrong shape for {name}"
                # Verify it's all ones (from TensorData)
                assert torch.allclose(tensor, torch.ones_like(tensor)), f"{name} should be all ones"
        
        print("✅ Tensor materialization: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Tensor materialization: FAILED - {e}")
        failed += 1
    
    # Test 3: Extract meta parameters
    try:
        meta_params = {arg.name: arg.value for arg in io_contract.args if arg.is_meta}
        
        assert "BLOCK_SIZE_M" in meta_params, "Missing BLOCK_SIZE_M"
        assert "BLOCK_SIZE_N" in meta_params, "Missing BLOCK_SIZE_N"
        assert meta_params["BLOCK_SIZE_M"] == 32, "Wrong value for BLOCK_SIZE_M"
        assert meta_params["BLOCK_SIZE_N"] == 64, "Wrong value for BLOCK_SIZE_N"
        
        print("✅ Meta parameters extraction: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Meta parameters extraction: FAILED - {e}")
        failed += 1
    
    # Test 4: Extract scalar arguments
    try:
        scalars = {arg.name: arg.value for arg in io_contract.args 
                  if arg.type in ["int", "float", "bool", "str"]}
        
        assert scalars["alpha"] == 1.5, "Wrong value for alpha"
        assert scalars["beta"] == 0.5, "Wrong value for beta"
        assert scalars["transpose"] == False, "Wrong value for transpose"
        
        print("✅ Scalar arguments extraction: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Scalar arguments extraction: FAILED - {e}")
        failed += 1
    
    print(f"\n📊 IOContract Integration Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 7: Edge Cases and Error Handling
# ============================================================================

async def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*80)
    print("🧪 TEST 7: Edge Cases and Error Handling")
    print("="*80)
    
    passed = 0
    failed = 0
    
    # Test 1: Very large tensor compression
    print("\n📝 Testing very large tensor compression...")
    try:
        large_tensor = torch.randn(1024, 1024)  # 4MB in float32
        
        # Without compression
        data_none = encode_tensor_to_data(large_tensor, compress="none")
        size_none = len(base64.b64decode(data_none.data_b64))
        
        # With compression
        data_zlib = encode_tensor_to_data(large_tensor, compress="zlib")
        size_zlib = len(base64.b64decode(data_zlib.data_b64))
        
        compression_ratio = size_none / size_zlib
        print(f"  Uncompressed: {size_none} bytes")
        print(f"  Compressed: {size_zlib} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        # Verify decompression works
        decoded = decode_tensor_from_data(data_zlib)
        assert torch.allclose(decoded.cpu(), large_tensor), "Large tensor mismatch after compression"
        
        print("✅ Large tensor compression: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Large tensor compression: FAILED - {e}")
        failed += 1
    
    # Test 2: Different devices
    if torch.cuda.is_available():
        print("\n📝 Testing cross-device tensor handling...")
        try:
            # CPU tensor encoded, decoded to CUDA
            cpu_tensor = torch.randn(32, 32)
            data = encode_tensor_to_data(cpu_tensor, compress="none")
            cuda_decoded = decode_tensor_from_data(data, device='cuda')
            
            assert cuda_decoded.device.type == 'cuda', "Tensor not on CUDA"
            assert torch.allclose(cuda_decoded.cpu(), cpu_tensor), "Cross-device values don't match"
            
            print("✅ Cross-device handling: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ Cross-device handling: FAILED - {e}")
            failed += 1
    
    # Test 3: Empty tensor handling
    print("\n📝 Testing empty tensor handling...")
    try:
        empty_tensor = torch.empty(0, 10)
        data = encode_tensor_to_data(empty_tensor)
        decoded = decode_tensor_from_data(data)
        
        assert decoded.shape == empty_tensor.shape, "Empty tensor shape mismatch"
        print("✅ Empty tensor handling: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Empty tensor handling: FAILED - {e}")
        failed += 1
    
    # Test 4: Deterministic generation with seeds
    print("\n📝 Testing deterministic generation...")
    try:
        spec1 = TensorSpec(shape=[100], dtype="float32", 
                          init=TensorInit(kind="randn", seed=999))
        spec2 = TensorSpec(shape=[100], dtype="float32",
                          init=TensorInit(kind="randn", seed=999))
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t1 = materialize_tensor(spec1, default_device=device)
        t2 = materialize_tensor(spec2, default_device=device)
        
        assert torch.equal(t1, t2), "Same seed should produce identical tensors"
        
        # Different seeds should produce different tensors
        spec3 = TensorSpec(shape=[100], dtype="float32",
                          init=TensorInit(kind="randn", seed=1000))
        t3 = materialize_tensor(spec3, default_device=device)
        
        assert not torch.equal(t1, t3), "Different seeds should produce different tensors"
        
        print("✅ Deterministic generation: PASSED")
        passed += 1
    except Exception as e:
        print(f"❌ Deterministic generation: FAILED - {e}")
        failed += 1
    
    print(f"\n📊 Edge Case Tests: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run all IOContract tests"""
    print("🚀 IOContract Test Suite")
    print("   Testing TensorData, TensorSpec, ArgSpec, LaunchConfig, and IOContract")
    
    # Check PyTorch availability
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available - some tests will be skipped")
    except ImportError:
        print("❌ PyTorch not installed! Install with: pip install torch")
        return
    
    tests = [
        ("TensorData Compression", test_tensor_data_compression),
        ("TensorInit", test_tensor_init),
        ("TensorSpec with Data", test_tensor_spec_with_data),
        ("ArgSpec", test_arg_spec),
        ("LaunchConfig", test_launch_config),
        ("IOContract Integration", test_io_contract_integration),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("IOContract components are fully functional:")
        print("  ✅ TensorData compression/decompression with zlib")
        print("  ✅ TensorSpec with TensorInit (randn, zeros, ones, uniform, full, arange)")
        print("  ✅ TensorSpec with literal TensorData")
        print("  ✅ ArgSpec for tensors, scalars, and meta parameters")
        print("  ✅ LaunchConfig for grid/block/Triton configurations")
        print("  ✅ Complete IOContract integration")
        print("  ✅ Edge cases and error handling")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        print("Review the output above for details")


if __name__ == "__main__":
    asyncio.run(main())
