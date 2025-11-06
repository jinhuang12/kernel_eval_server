#!/usr/bin/env python3
"""
Test script to verify the Pydantic migration and refactoring
"""

import sys
import json
from typing import Dict, Any

# Test imports
print("Testing imports...")

try:
    # Test constants import
    from shared.constants import DEFAULT_ATOL, DEFAULT_RTOL, DEFAULT_NUM_TRIALS
    print(f"✓ Constants imported: atol={DEFAULT_ATOL}, rtol={DEFAULT_RTOL}, trials={DEFAULT_NUM_TRIALS}")
except ImportError as e:
    print(f"✗ Failed to import constants: {e}")
    sys.exit(1)

try:
    # Test models import
    from shared.models import (
        KernelCode, IOContract, ArgSpec, TensorSpec, TensorInit,
        LaunchConfig, LaunchDim, RuntimeStats, KernelExecutionResult,
        CompareRequest, EvaluationRequest, KernelType
    )
    print("✓ All Pydantic models imported successfully")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

try:
    # Test utilities import
    from shared.request_validator import validate_num_trials, ValidationError
    from shared.response_utils import convert_result_to_dict, create_error_response
    from shared.io_contract_validator import validate_io_contract_dict
    print("✓ All utility modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import utilities: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("Testing Pydantic model migrations...")
print("="*60)


def test_tensor_spec():
    """Test TensorSpec Pydantic model"""
    print("\n1. Testing TensorSpec...")

    # Test creation
    tensor_spec = TensorSpec(
        shape=[1024, 1024],
        dtype="float32",
        init=TensorInit(kind="randn", seed=42)
    )

    # Test serialization
    dict_repr = tensor_spec.model_dump(exclude_none=True)
    assert dict_repr["shape"] == [1024, 1024]
    assert dict_repr["dtype"] == "float32"
    print("  ✓ TensorSpec creation and serialization")

    # Test backward compatibility
    old_dict = tensor_spec.to_dict()
    assert old_dict["shape"] == [1024, 1024]
    print("  ✓ Backward compatibility (to_dict)")

    # Test from_dict
    new_spec = TensorSpec.from_dict({
        "shape": [512],
        "dtype": "float16",
        "init": {"kind": "zeros"}
    })
    assert new_spec.shape == [512]
    assert new_spec.dtype == "float16"
    print("  ✓ Backward compatibility (from_dict)")


def test_io_contract():
    """Test IOContract Pydantic model"""
    print("\n2. Testing IOContract...")

    # Create IOContract
    io = IOContract(
        args=[
            ArgSpec(
                name="x",
                type="tensor",
                role="input",
                tensor_spec=TensorSpec(
                    shape=[1024],
                    dtype="float32",
                    init=TensorInit(kind="randn")
                )
            )
        ],
        launch=LaunchConfig(
            grid=LaunchDim(x=4),
            num_warps=4
        )
    )

    # Test serialization
    dict_repr = io.model_dump(exclude_none=True)
    assert len(dict_repr["args"]) == 1
    assert dict_repr["args"][0]["name"] == "x"
    print("  ✓ IOContract creation and serialization")

    # Test backward compatibility
    old_dict = io.to_dict()
    assert len(old_dict["args"]) == 1
    print("  ✓ Backward compatibility (to_dict)")


def test_kernel_code():
    """Test KernelCode Pydantic model"""
    print("\n3. Testing KernelCode...")

    # Create KernelCode
    kernel = KernelCode(
        source_code="def add(x, y): return x + y",
        kernel_type=KernelType.TORCH,
        metadata={"description": "Simple add kernel"}
    )

    # Test serialization
    dict_repr = kernel.model_dump(exclude_none=True)
    assert dict_repr["source_code"] == "def add(x, y): return x + y"
    assert dict_repr["kernel_type"] == KernelType.TORCH
    print("  ✓ KernelCode creation and serialization")

    # Test backward compatibility
    old_dict = kernel.to_dict()
    assert old_dict["kernel_type"] == "torch"  # Should be string
    print("  ✓ Backward compatibility (to_dict)")

    # Test from_dict
    new_kernel = KernelCode.from_dict({
        "source_code": "test",
        "kernel_type": "triton"
    })
    assert new_kernel.kernel_type == KernelType.TRITON
    print("  ✓ Backward compatibility (from_dict)")


def test_request_models():
    """Test request models with validators"""
    print("\n4. Testing Request Models...")

    # Test CompareRequest
    request_dict = {
        "ref_kernel": {
            "source_code": "def ref(): pass",
            "kernel_type": "torch"
        },
        "custom_kernel": {
            "source_code": "def custom(): pass",
            "kernel_type": "torch_cuda"
        }
    }

    compare_req = CompareRequest(**request_dict)
    assert isinstance(compare_req.ref_kernel, KernelCode)
    assert isinstance(compare_req.custom_kernel, KernelCode)
    assert compare_req.atol == DEFAULT_ATOL  # Should use constant
    assert compare_req.rtol == DEFAULT_RTOL  # Should use constant
    print("  ✓ CompareRequest with automatic KernelCode parsing")

    # Test EvaluationRequest
    eval_dict = {
        "kernel": {
            "source_code": "def test(): pass",
            "kernel_type": "torch"
        }
    }

    eval_req = EvaluationRequest(**eval_dict)
    assert isinstance(eval_req.kernel, KernelCode)
    assert eval_req.num_trials == DEFAULT_NUM_TRIALS  # Should use constant
    assert eval_req.timeout == DEFAULT_TIMEOUT  # Should use constant
    print("  ✓ EvaluationRequest with automatic KernelCode parsing")


def test_validation():
    """Test validation utilities"""
    print("\n5. Testing Validation...")

    # Test parameter validation
    try:
        validate_num_trials(-1)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "at least" in str(e)
        print("  ✓ Parameter validation catches invalid values")

    # Test IOContract validation
    io_dict = {
        "args": [
            {
                "name": "x",
                "type": "tensor",
                "role": "input",
                "tensor_spec": {
                    "shape": [1024],
                    "dtype": "float32",
                    "init": {"kind": "randn"}
                }
            }
        ],
        "launch": {"grid": {"x": 4}, "num_warps": 4}
    }

    is_valid, error = validate_io_contract_dict(io_dict, "triton")
    assert is_valid
    print("  ✓ IOContract validation for valid contract")

    # Test invalid IOContract
    invalid_io = {"args": [{"name": "x"}]}  # Missing required fields
    is_valid, error = validate_io_contract_dict(invalid_io, "triton")
    assert not is_valid
    assert "missing required 'type'" in error
    print("  ✓ IOContract validation catches missing fields")


def test_runtime_stats():
    """Test RuntimeStats Pydantic model"""
    print("\n6. Testing RuntimeStats...")

    stats = RuntimeStats(
        mean=10.5,
        std=0.5,
        min=9.8,
        max=11.2,
        median=10.4,
        percentile_95=10.9,
        percentile_99=11.1
    )

    # Test serialization
    dict_repr = stats.model_dump(exclude_none=True)
    assert dict_repr["mean"] == 10.5
    print("  ✓ RuntimeStats serialization")

    # Test backward compatibility
    old_dict = stats.to_dict()
    assert old_dict["mean"] == 10.5
    print("  ✓ RuntimeStats backward compatibility")


def main():
    """Run all tests"""
    print("\nRunning Pydantic migration tests...")

    try:
        test_tensor_spec()
        test_io_contract()
        test_kernel_code()
        test_request_models()
        test_validation()
        test_runtime_stats()

        print("\n" + "="*60)
        print("✅ All tests passed! The refactoring is working correctly.")
        print("="*60)

        print("\n📝 Summary of changes:")
        print("  • Migrated 20+ dataclasses to Pydantic BaseModel")
        print("  • Created shared utilities for validation and response handling")
        print("  • Fixed tolerance defaults mismatch (now using 1e-5 consistently)")
        print("  • Maintained backward compatibility with to_dict/from_dict")
        print("  • Request models now auto-parse nested KernelCode objects")

        print("\n🎯 Next steps:")
        print("  1. Replace mcp_server.py with mcp_server_refactored.py")
        print("  2. Update app.py to use shared utilities")
        print("  3. Run integration tests on EC2 instance")
        print("  4. Update client library if needed")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()