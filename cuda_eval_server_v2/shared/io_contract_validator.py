"""
IOContract validation for CUDA Evaluation Server V2
Provides comprehensive validation for IOContract structures with helpful error messages
"""

from typing import Dict, Any, Optional, Tuple, List
from shared.models import KernelType, IOContract


class IOContractValidationError(ValueError):
    """Custom exception for IOContract validation errors"""
    pass


def validate_io_contract(
    io_contract: Optional[IOContract],
    kernel_type: KernelType
) -> Tuple[bool, Optional[str]]:
    """
    Validate IOContract structure and provide helpful error messages.

    This comprehensive validation ensures that IOContracts have all required fields
    and that those fields contain valid values. It provides detailed error messages
    to help users fix their IOContract specifications.

    Args:
        io_contract: The IOContract object to validate (or None)
        kernel_type: Type of kernel (TORCH, TRITON, CUDA, etc.)

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Convert to dict if it's an IOContract object
    if io_contract and hasattr(io_contract, 'model_dump'):
        io_dict = io_contract.model_dump()
    elif io_contract and hasattr(io_contract, 'to_dict'):
        io_dict = io_contract.to_dict()
    elif isinstance(io_contract, dict):
        io_dict = io_contract
    else:
        io_dict = None

    return validate_io_contract_dict(io_dict, kernel_type.value if hasattr(kernel_type, 'value') else str(kernel_type))


def validate_io_contract_dict(io_dict: Optional[Dict[str, Any]], kernel_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate IOContract dictionary structure and provide helpful error messages.

    Args:
        io_dict: Dictionary representation of IO contract
        kernel_type: Type of kernel as string (torch, triton, cuda, etc.)

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    # Check if IOContract is required for this kernel type
    requires_io_contract = kernel_type.lower() in ["triton", "cuda", "multi_kernel"]

    if not io_dict:
        if requires_io_contract:
            return False, (
                f"IOContract is REQUIRED for {kernel_type.upper()} kernels. "
                "Must specify 'args' array and 'launch' configuration. "
                "See documentation for examples."
            )
        return True, None  # Optional for TORCH/TORCH_CUDA

    # Validate 'args' field
    if "args" not in io_dict:
        return False, (
            "Missing required 'args' field in IOContract. "
            "Must be an array of argument specifications. "
            "Example: {'args': [{'name': 'x', 'type': 'tensor', 'role': 'input', 'tensor_spec': {...}}]}"
        )

    args = io_dict.get("args", [])
    if not isinstance(args, list):
        return False, "'args' must be an array of argument specifications"

    # Validate each argument
    for i, arg in enumerate(args):
        is_valid, error_msg = validate_arg_spec(arg, i)
        if not is_valid:
            return False, error_msg

    # Validate launch configuration for kernels that need it
    if requires_io_contract and "launch" not in io_dict:
        return False, (
            f"IOContract for {kernel_type.upper()} kernel missing 'launch' configuration. "
            "For Triton: {'grid': {'x': N}, 'num_warps': 4}. "
            "For CUDA: {'grid': {'x': N}, 'block': {'x': 256}}"
        )

    # Validate launch configuration if present
    if "launch" in io_dict:
        is_valid, error_msg = validate_launch_config(io_dict["launch"], kernel_type)
        if not is_valid:
            return False, error_msg

    return True, None


def validate_arg_spec(arg: Any, index: int) -> Tuple[bool, Optional[str]]:
    """
    Validate a single argument specification.

    Args:
        arg: The argument specification to validate
        index: Index of the argument in the args array

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(arg, dict):
        return False, f"Argument {index} must be an object/dictionary"

    # Check required fields
    if "name" not in arg:
        return False, f"Argument {index}: missing required 'name' field"

    arg_name = arg.get("name", f"arg_{index}")

    if "type" not in arg:
        return False, (
            f"Argument '{arg_name}': missing required 'type' field. "
            "Must be one of: 'tensor', 'int', 'float', 'str', 'bool'"
        )

    arg_type = arg.get("type")
    valid_types = ["tensor", "int", "float", "str", "bool"]
    if arg_type not in valid_types:
        return False, (
            f"Argument '{arg_name}': invalid type '{arg_type}'. "
            f"Must be one of: {', '.join(valid_types)}"
        )

    if "role" not in arg:
        return False, (
            f"Argument '{arg_name}': missing required 'role' field. "
            "Must be one of: 'input', 'output', 'inout'"
        )

    role = arg.get("role")
    valid_roles = ["input", "output", "inout"]
    if role not in valid_roles:
        return False, (
            f"Argument '{arg_name}': invalid role '{role}'. "
            f"Must be one of: {', '.join(valid_roles)}"
        )

    # Validate tensor-specific fields
    if arg_type == "tensor":
        return validate_tensor_arg(arg, arg_name)

    # Validate scalar values
    elif arg_type in ["int", "float", "str", "bool"]:
        return validate_scalar_arg(arg, arg_name, arg_type)

    return True, None


def validate_tensor_arg(arg: Dict[str, Any], arg_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a tensor argument specification.

    Args:
        arg: The tensor argument specification
        arg_name: Name of the argument for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if "tensor_spec" not in arg:
        return False, (
            f"Tensor argument '{arg_name}': missing required 'tensor_spec'. "
            "Must include 'shape' (array), 'dtype' (string), and optionally 'init' (for input tensors). "
            "Example: {'shape': [1024], 'dtype': 'float32', 'init': {'kind': 'randn', 'seed': 42}}"
        )

    tensor_spec = arg.get("tensor_spec", {})

    if "shape" not in tensor_spec:
        return False, f"Tensor '{arg_name}': tensor_spec missing 'shape' (array of integers)"

    shape = tensor_spec.get("shape")
    if not isinstance(shape, list):
        return False, f"Tensor '{arg_name}': shape must be an array of integers"

    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return False, f"Tensor '{arg_name}': shape dimensions must be positive integers"

    if "dtype" not in tensor_spec:
        return False, (
            f"Tensor '{arg_name}': tensor_spec missing 'dtype'. "
            "Valid dtypes: float32, float64, float16, bfloat16, int32, int64, int8, uint8, bool"
        )

    dtype = tensor_spec.get("dtype")
    valid_dtypes = [
        "float32", "float64", "float16", "bfloat16",
        "int32", "int64", "int16", "int8", "uint8", "bool"
    ]
    if dtype not in valid_dtypes:
        return False, (
            f"Tensor '{arg_name}': invalid dtype '{dtype}'. "
            f"Valid dtypes: {', '.join(valid_dtypes)}"
        )

    # Check init for input/inout tensors
    role = arg.get("role")
    if role in ["input", "inout"] and "init" not in tensor_spec:
        return False, (
            f"Tensor '{arg_name}' with role='{role}': tensor_spec missing 'init'. "
            "Input tensors need initialization method. "
            "Example: {'kind': 'randn', 'seed': 42} or {'kind': 'zeros'}"
        )

    # Validate init if present
    if "init" in tensor_spec:
        is_valid, error_msg = validate_tensor_init(tensor_spec["init"], arg_name)
        if not is_valid:
            return False, error_msg

    return True, None


def validate_tensor_init(init: Dict[str, Any], arg_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate tensor initialization specification.

    Args:
        init: The initialization specification
        arg_name: Name of the tensor argument for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if "kind" not in init:
        return False, (
            f"Tensor '{arg_name}': init missing 'kind'. "
            "Valid kinds: randn, uniform, zeros, ones, full, arange"
        )

    kind = init["kind"]
    valid_kinds = ["randn", "uniform", "zeros", "ones", "full", "arange"]
    if kind not in valid_kinds:
        return False, (
            f"Tensor '{arg_name}': invalid init kind '{kind}'. "
            f"Valid kinds: {', '.join(valid_kinds)}"
        )

    # Validate kind-specific parameters
    if kind == "full" and "fill_value" not in init:
        return False, f"Tensor '{arg_name}': init kind='full' requires 'fill_value' parameter"

    if kind == "uniform":
        # Low and high are optional (default to 0 and 1)
        if "low" in init and "high" in init:
            if init["low"] >= init["high"]:
                return False, f"Tensor '{arg_name}': uniform init requires low < high"

    if kind == "arange":
        # Step is optional (defaults to 1)
        if "step" in init and init["step"] == 0:
            return False, f"Tensor '{arg_name}': arange init step cannot be zero"

    return True, None


def validate_scalar_arg(arg: Dict[str, Any], arg_name: str, arg_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a scalar argument specification.

    Args:
        arg: The scalar argument specification
        arg_name: Name of the argument for error messages
        arg_type: Type of the scalar (int, float, str, bool)

    Returns:
        Tuple of (is_valid, error_message)
    """
    is_meta = arg.get("is_meta", False)

    if not is_meta and "value" not in arg:
        return False, (
            f"Scalar argument '{arg_name}' (type={arg_type}): missing 'value'. "
            "Scalars must have 'value' field unless is_meta=true (Triton only)"
        )

    if "value" in arg:
        value = arg["value"]
        # Type check the value
        if arg_type == "int" and not isinstance(value, int):
            return False, f"Scalar '{arg_name}': value must be an integer"
        elif arg_type == "float" and not isinstance(value, (int, float)):
            return False, f"Scalar '{arg_name}': value must be a number"
        elif arg_type == "str" and not isinstance(value, str):
            return False, f"Scalar '{arg_name}': value must be a string"
        elif arg_type == "bool" and not isinstance(value, bool):
            return False, f"Scalar '{arg_name}': value must be a boolean"

    return True, None


def validate_launch_config(launch: Dict[str, Any], kernel_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate launch configuration for a kernel.

    Args:
        launch: The launch configuration dictionary
        kernel_type: Type of kernel (triton, cuda, etc.)

    Returns:
        Tuple of (is_valid, error_message)
    """
    kernel_type_lower = kernel_type.lower()

    # Check for grid configuration (required for all)
    if "grid" not in launch:
        return False, "Launch configuration missing required 'grid' field"

    grid = launch["grid"]
    if not isinstance(grid, dict):
        return False, "Launch 'grid' must be an object with x, y, z dimensions"

    # Validate grid dimensions
    for dim in ["x", "y", "z"]:
        if dim in grid:
            if not isinstance(grid[dim], int) or grid[dim] <= 0:
                return False, f"Grid dimension '{dim}' must be a positive integer"

    if "x" not in grid:
        return False, "Grid must have at least 'x' dimension"

    # CUDA-specific validation
    if kernel_type_lower == "cuda":
        if "block" not in launch:
            return False, (
                "CUDA kernel launch configuration missing 'block' field. "
                "Example: {'grid': {'x': 100}, 'block': {'x': 256}}"
            )

        block = launch["block"]
        if not isinstance(block, dict):
            return False, "Launch 'block' must be an object with x, y, z dimensions"

        # Validate block dimensions
        for dim in ["x", "y", "z"]:
            if dim in block:
                if not isinstance(block[dim], int) or block[dim] <= 0:
                    return False, f"Block dimension '{dim}' must be a positive integer"

        if "x" not in block:
            return False, "Block must have at least 'x' dimension"

        # Check block size limits (typical CUDA limits)
        total_threads = block.get("x", 1) * block.get("y", 1) * block.get("z", 1)
        if total_threads > 1024:
            return False, f"Total threads per block ({total_threads}) exceeds CUDA limit of 1024"

    # Triton-specific validation
    elif kernel_type_lower == "triton":
        if "num_warps" in launch:
            num_warps = launch["num_warps"]
            if not isinstance(num_warps, int) or num_warps <= 0:
                return False, "num_warps must be a positive integer"
            if num_warps not in [1, 2, 4, 8, 16, 32]:
                return False, "num_warps must be a power of 2 between 1 and 32"

    return True, None


def raise_if_invalid(io_contract: Optional[IOContract], kernel_type: KernelType) -> None:
    """
    Validate IOContract and raise exception if invalid.

    This is a convenience function that raises an exception instead of
    returning a tuple, useful for FastAPI/MCP validation.

    Args:
        io_contract: The IOContract to validate
        kernel_type: Type of kernel

    Raises:
        IOContractValidationError: If validation fails
    """
    is_valid, error_msg = validate_io_contract(io_contract, kernel_type)
    if not is_valid:
        raise IOContractValidationError(error_msg)