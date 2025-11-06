"""
Unit tests for IOContract parsing and validation
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import IOContract, ArgSpec, TensorSpec, TensorInit, LaunchConfig, LaunchDim
from io_contract.manager import IOContractManager


@pytest.mark.unit
class TestIOContract:
    """Tests for IOContract model"""
    
    def test_create_simple_io_contract(self):
        """Test creating a simple IOContract"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=[100], dtype="float32"), role="input"),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(shape=[100], dtype="float32"), role="output")
            ]
        )
        
        assert len(io_contract.args) == 2
        assert io_contract.args[0].name == "x"
        assert io_contract.args[0].role == "input"
        assert io_contract.args[1].name == "y"
        assert io_contract.args[1].role == "output"
    
    def test_io_contract_with_init_values(self):
        """Test IOContract with initialization values"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(
                    shape=[10, 10],
                    dtype="float32",
                    init=TensorInit(kind="randn")
                ), role="input"),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(
                    shape=[10, 10],
                    dtype="float32",
                    init=TensorInit(kind="ones")
                ), role="input"),
                ArgSpec(name="out", type="tensor", tensor_spec=TensorSpec(shape=[10, 10], dtype="float32"), role="output")
            ]
        )
        
        assert io_contract.args[0].tensor_spec.init.kind == "randn"
        assert io_contract.args[1].tensor_spec.init.kind == "ones"
    
    def test_io_contract_with_launch_config(self):
        """Test IOContract with launch configuration"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=[1024], dtype="float32"), role="input"),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(shape=[1024], dtype="float32"), role="output")
            ],
            launch=LaunchConfig(
                grid=LaunchDim(x=4, y=1, z=1),
                block=LaunchDim(x=256, y=1, z=1)
            )
        )
        
        assert io_contract.launch is not None
        assert io_contract.launch.grid.x == 4
        assert io_contract.launch.block.x == 256
    
    def test_io_contract_serialization(self):
        """Test IOContract serialization/deserialization"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=[100], dtype="float32"), role="input"),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(shape=[100], dtype="float32"), role="output")
            ]
        )
        
        # Serialize to dict
        data = io_contract.to_dict()
        
        # Deserialize back
        io_contract2 = IOContract.from_dict(data)
        
        assert io_contract2.args[0].name == "x"
        assert io_contract2.args[1].name == "y"
    
    def test_dynamic_shapes(self):
        """Test IOContract with dynamic shapes"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=["batch", 128], dtype="float32"), role="input"),
                ArgSpec(name="batch", type="int", value=32),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(shape=["batch", 128], dtype="float32"), role="output")
            ]
        )
        
        assert io_contract.args[0].tensor_spec.shape[0] == "batch"
        assert io_contract.args[1].value == 32


@pytest.mark.unit
@pytest.mark.gpu
class TestIOContractManager:
    """Tests for IOContract manager"""
    
    def test_generate_inputs_from_contract(self):
        """Test generating inputs from IOContract"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(
                    shape=[10, 10],
                    dtype="float32",
                    init=TensorInit(kind="randn")
                ), role="input"),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(
                    shape=[10, 10],
                    dtype="float32",
                    init=TensorInit(kind="ones")
                ), role="input"),
                ArgSpec(name="out", type="tensor", tensor_spec=TensorSpec(shape=[10, 10], dtype="float32"), role="output")
            ]
        )
        
        manager = IOContractManager()
        inputs = manager.generate_inputs(io_contract)
        
        assert "x" in inputs
        assert "y" in inputs
        assert inputs["x"].shape == torch.Size([10, 10])
        assert inputs["y"].shape == torch.Size([10, 10])
        assert torch.allclose(inputs["y"], torch.ones(10, 10))
    
    def test_generate_with_scalar_inputs(self):
        """Test generating mixed tensor and scalar inputs"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(shape=[100], dtype="float32"), role="input"),
                ArgSpec(name="n", type="int", value=100),
                ArgSpec(name="y", type="tensor", tensor_spec=TensorSpec(shape=[100], dtype="float32"), role="output")
            ]
        )
        
        manager = IOContractManager()
        inputs = manager.generate_inputs(io_contract)
        
        assert "x" in inputs
        assert "n" in inputs
        assert isinstance(inputs["x"], torch.Tensor)
        assert isinstance(inputs["n"], int)
        assert inputs["n"] == 100
    
    def test_validate_outputs(self):
        """Test validating outputs against contract"""
        io_contract = IOContract(
            args=[
                ArgSpec(name="out", type="tensor", tensor_spec=TensorSpec(shape=[10, 10], dtype="float32"), role="output")
            ]
        )
        
        manager = IOContractManager()
        
        # Valid output
        valid_output = torch.randn(10, 10)
        assert manager.validate_output(valid_output, io_contract)
        
        # Invalid shape
        invalid_output = torch.randn(5, 5)
        assert not manager.validate_output(invalid_output, io_contract)
    
    def test_different_init_methods(self):
        """Test different tensor initialization methods"""
        test_cases = [
            (TensorInit(kind="zeros"), torch.zeros),
            (TensorInit(kind="ones"), torch.ones),
            (TensorInit(kind="randn"), torch.randn),
            (TensorInit(kind="uniform"), torch.rand),
        ]
        
        manager = IOContractManager()
        
        for init_method, expected_fn in test_cases:
            io_contract = IOContract(
                args=[
                    ArgSpec(name="x", type="tensor", tensor_spec=TensorSpec(
                        shape=[5, 5],
                        dtype="float32",
                        init=init_method
                    ), role="input")
                ]
            )
            
            inputs = manager.generate_inputs(io_contract)
            
            if init_method.kind in ["zeros", "ones"]:
                expected = expected_fn(5, 5)
                assert torch.allclose(inputs["x"], expected)
            else:
                # For random, just check shape and dtype
                assert inputs["x"].shape == torch.Size([5, 5])
                assert inputs["x"].dtype == torch.float32