"""
Args generator with strategy pattern for different kernel types
Replaces the old input_generator with a more flexible approach
"""

import ast
import torch
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import numpy as np

from .models import KernelCode, KernelType, ArgSpec, TensorSpec

logger = logging.getLogger(__name__)


class BaseArgsGeneratorStrategy(ABC):
    """Base class for argument generation strategies"""
    
    @abstractmethod
    def generate_args(
        self,
        kernel_code: KernelCode,
        input_specs: Optional[List[ArgSpec]],
        output_spec: Optional[TensorSpec],
        device: torch.device
    ) -> Tuple[List[Any], Any, Optional[torch.nn.Module]]:
        """
        Generate input arguments and expected output for kernel
        
        Args:
            kernel_code: Kernel source code and type
            input_specs: Optional user-provided input specifications
            output_spec: Optional user-provided output specification
            device: CUDA device to use
            
        Returns:
            Tuple of (inputs, expected_output, optional_model_instance)
        """
        pass


class TorchCudaArgsStrategy(BaseArgsGeneratorStrategy):
    """Use KernelBench logic for torch_cuda kernels - auto-generates from code"""
    
    def generate_args(
        self,
        kernel_code: KernelCode,
        input_specs: Optional[List[ArgSpec]],
        output_spec: Optional[TensorSpec],
        device: torch.device
    ) -> Tuple[List[Any], Any, Optional[torch.nn.Module]]:
        """Generate args using KernelBench's existing logic"""
        try:
            # Import KernelBench functions
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            kb_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            if kb_dir not in sys.path:
                sys.path.insert(0, kb_dir)
            
            from KernelBench.eval import load_original_model_and_inputs, set_seed
            
            logger.debug("Generating model inputs using KernelBench logic")
            
            context = {}
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                kernel_code.source_code, context
            )
            
            # Use fixed seed for deterministic input generation
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            
            # Move inputs to target device
            inputs = [
                x.cuda(device) if isinstance(x, torch.Tensor) else x 
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device) if isinstance(x, torch.Tensor) else x 
                for x in init_inputs
            ]
            
            logger.debug(f"Generated {len(inputs)} inputs and {len(init_inputs)} init inputs")
            
            # Create and initialize model
            model = Model(*init_inputs).cuda(device)
            
            # Generate reference output
            with torch.no_grad():
                set_seed(42)
                torch.cuda.synchronize(device)
                reference_output = model(*inputs)
                torch.cuda.synchronize(device)
            
            logger.debug(f"Generated reference output with shape: {reference_output.shape if hasattr(reference_output, 'shape') else type(reference_output)}")
            
            return inputs, reference_output, model
            
        except Exception as e:
            logger.error(f"Failed to generate model inputs: {e}")
            raise RuntimeError(f"Input generation failed: {str(e)}")


class ManualArgsStrategy(BaseArgsGeneratorStrategy):
    """Generate args from user-provided specifications"""
    
    def generate_args(
        self,
        kernel_code: KernelCode,
        input_specs: Optional[List[ArgSpec]],
        output_spec: Optional[TensorSpec],
        device: torch.device
    ) -> Tuple[List[Any], Any, None]:
        """Generate args from user specifications"""
        
        if not input_specs:
            raise ValueError("Manual args strategy requires input_specs to be provided")
        
        inputs = []
        
        for spec in input_specs:
            if spec.type == "tensor":
                if not spec.tensor_spec:
                    raise ValueError(f"Tensor argument '{spec.name}' missing tensor_spec")
                
                # Create tensor from specification
                tensor = self._create_tensor(spec.tensor_spec, device)
                inputs.append(tensor)
                
            elif spec.type in ["int", "float", "str", "bool"]:
                if spec.value is None:
                    # Generate default value based on type
                    if spec.type == "int":
                        inputs.append(1)
                    elif spec.type == "float":
                        inputs.append(1.0)
                    elif spec.type == "str":
                        inputs.append("")
                    elif spec.type == "bool":
                        inputs.append(True)
                else:
                    inputs.append(spec.value)
            else:
                raise ValueError(f"Unsupported argument type: {spec.type}")
        
        # Generate expected output tensor if specified
        expected_output = None
        if output_spec:
            expected_output = self._create_tensor(output_spec, device)
        
        logger.debug(f"Generated {len(inputs)} manual inputs")
        return inputs, expected_output, None
    
    def _create_tensor(self, tensor_spec: TensorSpec, device: torch.device) -> torch.Tensor:
        """Create a tensor from specification"""
        
        # Map dtype strings to torch dtypes
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        
        dtype = dtype_map.get(tensor_spec.dtype, torch.float32)
        
        # Generate random tensor with appropriate values for dtype
        if dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            # Floating point: use random normal distribution
            tensor = torch.randn(tensor_spec.shape, dtype=dtype, device=device)
        elif dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
            # Signed integers: use random integers in reasonable range
            tensor = torch.randint(-100, 100, tensor_spec.shape, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            # Unsigned byte: 0-255
            tensor = torch.randint(0, 256, tensor_spec.shape, dtype=dtype, device=device)
        elif dtype == torch.bool:
            # Boolean: random true/false
            tensor = torch.randint(0, 2, tensor_spec.shape, dtype=dtype, device=device).bool()
        else:
            # Fallback to zeros
            tensor = torch.zeros(tensor_spec.shape, dtype=dtype, device=device)
        
        return tensor


class TritonArgsStrategy(BaseArgsGeneratorStrategy):
    """Enhanced Triton-specific argument generator with auto-detection and validation"""
    
    def __init__(self):
        self.logger = logger
        # Map of Triton to PyTorch dtypes
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "int16": torch.int16,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        
    def generate_args(
        self,
        kernel_code: KernelCode,
        input_specs: Optional[List[ArgSpec]],
        output_spec: Optional[TensorSpec],
        device: torch.device
    ) -> Tuple[List[Any], Any, Optional[torch.nn.Module]]:
        """
        Generate args for Triton kernel with auto-detection fallback
        
        Args:
            kernel_code: Triton kernel source code
            input_specs: Optional user-provided input specifications
            output_spec: Optional user-provided output specification
            device: CUDA device to use
            
        Returns:
            Tuple of (inputs, expected_output, optional_model_instance)
        """
        # If user provided specs, use them
        if input_specs:
            return self._generate_from_specs(input_specs, output_spec, device)
        
        # Try auto-detection from kernel source
        try:
            detected_specs, detected_output = self._auto_detect_io_contract(kernel_code)
            if detected_specs:
                return self._generate_from_specs(detected_specs, detected_output, device)
        except Exception as e:
            self.logger.warning(f"Auto-detection failed: {e}, using defaults")
        
        # Fallback to default inputs
        return self._generate_default_inputs(device)
    
    def _generate_from_specs(
        self,
        input_specs: List[ArgSpec],
        output_spec: Optional[TensorSpec],
        device: torch.device
    ) -> Tuple[List[Any], Any, Optional[torch.nn.Module]]:
        """Generate inputs from specifications"""
        inputs = []
        
        for spec in input_specs:
            if spec.type == "tensor":
                if not spec.tensor_spec:
                    # Default tensor spec
                    tensor_spec = TensorSpec(shape=[1024], dtype="float32")
                else:
                    tensor_spec = spec.tensor_spec
                
                # Create tensor with appropriate values
                tensor = self._create_tensor(tensor_spec, device)
                inputs.append(tensor)
                
            elif spec.type in ["int", "float", "bool"]:
                # Scalar values
                if spec.value is not None:
                    inputs.append(spec.value)
                else:
                    # Default values
                    if spec.type == "int":
                        inputs.append(256)  # Common block size
                    elif spec.type == "float":
                        inputs.append(1.0)
                    else:
                        inputs.append(True)
            else:
                self.logger.warning(f"Unknown argument type: {spec.type}")
                inputs.append(None)
        
        # Generate expected output if specified
        expected_output = None
        if output_spec:
            expected_output = self._create_tensor(output_spec, device)
        elif inputs and isinstance(inputs[0], torch.Tensor):
            # Default: output same shape as first input
            expected_output = torch.empty_like(inputs[0])
        
        return inputs, expected_output, None
    
    def _create_tensor(self, tensor_spec: TensorSpec, device: torch.device) -> torch.Tensor:
        """Create a tensor from specification with Triton-appropriate initialization"""
        dtype = self.dtype_map.get(tensor_spec.dtype, torch.float32)
        
        # Handle dynamic shapes (None or -1 becomes concrete size)
        shape = []
        for dim in tensor_spec.shape:
            if dim is None or dim == -1:
                shape.append(1024)  # Default size for dynamic dimensions
            else:
                shape.append(dim)
        
        # Generate tensor with appropriate initialization
        if dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
            # Floating point: use randn for better numerical properties
            tensor = torch.randn(shape, dtype=dtype, device=device)
        elif dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
            # Integers: reasonable range
            tensor = torch.randint(-100, 100, shape, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            # Unsigned byte
            tensor = torch.randint(0, 256, shape, dtype=dtype, device=device)
        elif dtype == torch.bool:
            # Boolean
            tensor = torch.randint(0, 2, shape, dtype=dtype, device=device).bool()
        else:
            # Fallback
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        
        # Ensure contiguous for better performance
        return tensor.contiguous()
    
    def _auto_detect_io_contract(self, kernel_code: KernelCode) -> Tuple[List[ArgSpec], Optional[TensorSpec]]:
        """Auto-detect IO contract from Triton kernel source"""
        import ast
        import re
        
        input_specs = []
        output_spec = None
        
        try:
            # Parse the source code
            tree = ast.parse(kernel_code.source_code)
            
            # Find @triton.jit decorated function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for triton.jit decorator
                    has_jit = any(
                        self._is_triton_jit_decorator(dec) 
                        for dec in node.decorator_list
                    )
                    
                    if has_jit:
                        # Extract parameters
                        for arg in node.args.args:
                            arg_name = arg.arg
                            
                            # Detect tensor pointers
                            if arg_name.endswith('_ptr'):
                                base_name = arg_name[:-4]
                                
                                # Default tensor spec
                                tensor_spec = TensorSpec(
                                    shape=[None],  # Dynamic shape
                                    dtype="float32"
                                )
                                
                                input_specs.append(ArgSpec(
                                    name=base_name,
                                    type="tensor",
                                    tensor_spec=tensor_spec
                                ))
                            
                            # Check for constexpr annotation
                            elif arg.annotation and 'constexpr' in str(arg.annotation):
                                # Constexpr parameters are compile-time constants
                                continue
                            
                            # Other parameters (likely scalars)
                            else:
                                # Try to infer type from name
                                if any(kw in arg_name.lower() for kw in ['size', 'num', 'count', 'stride']):
                                    input_specs.append(ArgSpec(
                                        name=arg_name,
                                        type="int",
                                        value=None
                                    ))
                        
                        break
            
            # Infer output spec (usually similar to input for element-wise ops)
            if input_specs:
                # Find first tensor spec for output shape
                for spec in input_specs:
                    if spec.type == "tensor" and spec.tensor_spec:
                        output_spec = TensorSpec(
                            shape=spec.tensor_spec.shape,
                            dtype=spec.tensor_spec.dtype
                        )
                        break
            
        except Exception as e:
            self.logger.debug(f"Failed to parse Triton kernel: {e}")
        
        return input_specs, output_spec
    
    def _is_triton_jit_decorator(self, decorator: ast.AST) -> bool:
        """Check if a decorator is @triton.jit"""
        if isinstance(decorator, ast.Name):
            return decorator.id == 'jit'
        elif isinstance(decorator, ast.Attribute):
            if isinstance(decorator.value, ast.Name):
                return decorator.value.id == 'triton' and decorator.attr == 'jit'
        elif isinstance(decorator, ast.Call):
            if hasattr(decorator, 'func'):
                return self._is_triton_jit_decorator(decorator.func)
        return False
    
    def _generate_default_inputs(self, device: torch.device) -> Tuple[List[Any], Any, None]:
        """Generate default inputs when no specs available"""
        # Default: two float32 tensors of size 1024
        input1 = torch.randn(1024, dtype=torch.float32, device=device)
        input2 = torch.randn(1024, dtype=torch.float32, device=device)
        
        # Expected output same shape
        expected_output = torch.empty_like(input1)
        
        return [input1, input2], expected_output, None


class ArgsGeneratorService:
    """Service for selecting and using appropriate args generation strategy"""
    
    def __init__(self):
        """Initialize with strategy registry"""
        self.strategies: Dict[KernelType, BaseArgsGeneratorStrategy] = {
            KernelType.TORCH_CUDA: TorchCudaArgsStrategy(),
            KernelType.TORCH: TorchCudaArgsStrategy(),  # PyTorch uses same strategy
            KernelType.CUDA: ManualArgsStrategy(),
            KernelType.TRITON: TritonArgsStrategy(),
        }
    
    def generate_args(
        self,
        kernel_code: KernelCode,
        input_specs: Optional[List[ArgSpec]],
        output_spec: Optional[TensorSpec],
        device: torch.device
    ) -> Tuple[List[Any], Any, Optional[torch.nn.Module]]:
        """
        Generate arguments using appropriate strategy
        
        Args:
            kernel_code: Kernel code with type information
            input_specs: Optional user-provided input specifications
            output_spec: Optional user-provided output specification
            device: CUDA device to use
            
        Returns:
            Tuple of (inputs, expected_output, optional_model_instance)
        """
        
        # If user provided specs, use manual strategy regardless of kernel type
        if input_specs is not None and len(input_specs) > 0:
            logger.info(f"Using manual args strategy for {kernel_code.kernel_type} (user provided specs)")
            strategy = ManualArgsStrategy()
        else:
            # Select strategy based on kernel type
            strategy = self.strategies.get(kernel_code.kernel_type)
            if not strategy:
                raise ValueError(f"No args generation strategy for kernel type: {kernel_code.kernel_type}")
            
            logger.info(f"Using {strategy.__class__.__name__} for {kernel_code.kernel_type} kernel")
        
        return strategy.generate_args(kernel_code, input_specs, output_spec, device)


# Backward compatibility wrapper
class ModelInputGenerator:
    """Backward compatibility wrapper for old input_generator usage"""
    
    @staticmethod
    def generate_model_inputs_and_outputs(ref_code: str, device: torch.device):
        """Legacy method for backward compatibility"""
        
        # Create KernelCode wrapper
        kernel_code = KernelCode(
            source_code=ref_code,
            kernel_type=KernelType.TORCH_CUDA  # Assume torch_cuda for legacy
        )
        
        # Use new service
        service = ArgsGeneratorService()
        inputs, reference_output, model = service.generate_args(
            kernel_code, None, None, device
        )
        
        # Return in legacy format (with init_inputs extracted from model)
        init_inputs = []  # Legacy format didn't separate these well
        return inputs, init_inputs, reference_output, model
