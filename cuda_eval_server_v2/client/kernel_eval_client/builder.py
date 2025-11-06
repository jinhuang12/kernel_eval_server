"""
IOContract Builder - Fluent interface for building IOContracts
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

# Add parent directories to path to import from shared
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import (
    ArgSpec, IOContract, LaunchConfig, LaunchDim, TensorSpec
)


class IOContractBuilder:
    """
    Fluent interface for building IOContracts.
    
    Example:
        contract = IOContractBuilder() \\
            .add_input_tensor("x", create_randn_spec([1024, 512], seed=42)) \\
            .add_input_tensor("y", create_ones_spec([512, 256])) \\
            .add_output_tensor("result", [1024, 256], "float32") \\
            .add_scalar("alpha", "float", 2.0) \\
            .add_meta_param("BLOCK_SIZE", 256) \\
            .set_launch_config(launch_config) \\
            .build()
    """
    
    def __init__(self):
        self.args: List[ArgSpec] = []
        self.launch_config: Optional[LaunchConfig] = None
    
    def add_input_tensor(self, name: str, tensor_spec: TensorSpec) -> "IOContractBuilder":
        """Add an input tensor argument."""
        self.args.append(ArgSpec(
            name=name,
            type="tensor",
            tensor_spec=tensor_spec,
            role="input"
        ))
        return self
    
    def add_output_tensor(self, name: str, shape: List[int], dtype: str = "float32") -> "IOContractBuilder":
        """Add an output tensor argument."""
        tensor_spec = TensorSpec(shape=shape, dtype=dtype)
        self.args.append(ArgSpec(
            name=name,
            type="tensor",
            tensor_spec=tensor_spec,
            role="output"
        ))
        return self
    
    def add_inout_tensor(self, name: str, tensor_spec: TensorSpec) -> "IOContractBuilder":
        """Add an input/output tensor argument."""
        self.args.append(ArgSpec(
            name=name,
            type="tensor",
            tensor_spec=tensor_spec,
            role="inout"
        ))
        return self
    
    def add_scalar(self, name: str, scalar_type: str, value: Union[int, float, str, bool]) -> "IOContractBuilder":
        """
        Add a scalar argument.
        
        Args:
            name: Argument name
            scalar_type: Type ("int", "float", "str", "bool")
            value: Scalar value
        """
        self.args.append(ArgSpec(
            name=name,
            type=scalar_type,
            value=value,
            role="input"
        ))
        return self
    
    def add_meta_param(self, name: str, value: int) -> "IOContractBuilder":
        """
        Add a meta parameter (Triton constexpr).
        
        Args:
            name: Parameter name
            value: Integer value
        """
        self.args.append(ArgSpec(
            name=name,
            type="int",
            value=value,
            role="input",
            is_meta=True
        ))
        return self
    
    def set_launch_config(self, launch_config: LaunchConfig) -> "IOContractBuilder":
        """Set the kernel launch configuration."""
        self.launch_config = launch_config
        return self
    
    def set_grid(self, x: int, y: int = 1, z: int = 1) -> "IOContractBuilder":
        """Set grid dimensions (convenience method)."""
        if self.launch_config is None:
            self.launch_config = LaunchConfig()
        self.launch_config.grid = LaunchDim(x=x, y=y, z=z)
        return self
    
    def set_block(self, x: int, y: int = 1, z: int = 1) -> "IOContractBuilder":
        """Set block dimensions for CUDA kernels (convenience method)."""
        if self.launch_config is None:
            self.launch_config = LaunchConfig()
        self.launch_config.block = LaunchDim(x=x, y=y, z=z)
        return self
    
    def set_num_warps(self, num_warps: int) -> "IOContractBuilder":
        """Set number of warps for Triton kernels (convenience method)."""
        if self.launch_config is None:
            self.launch_config = LaunchConfig()
        self.launch_config.num_warps = num_warps
        return self
    
    def set_num_stages(self, num_stages: int) -> "IOContractBuilder":
        """Set number of pipeline stages for Triton kernels (convenience method)."""
        if self.launch_config is None:
            self.launch_config = LaunchConfig()
        self.launch_config.num_stages = num_stages
        return self
    
    def build(self) -> IOContract:
        """Build the final IOContract."""
        return IOContract(
            args=self.args,
            launch=self.launch_config
        )