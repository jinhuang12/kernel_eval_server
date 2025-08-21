"""
Executable kernel implementations for different kernel types
Refactored from TorchCudaModelWrapper to use BaseExecutableKernel interface
"""

import logging
import torch
import ast
from typing import Optional, Any, List, Dict

from shared.models import BaseExecutableKernel, IOContract, KernelCode, KernelType, CompiledKernelInfo
from shared.utils import materialize_tensor
from KernelBench.eval import load_original_model_and_inputs, set_seed

logger = logging.getLogger(__name__)
# Handle CuPy import
try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cupy = None


class TorchCudaExecutableKernel(BaseExecutableKernel):
    """TORCH_CUDA specific executable kernel with C++ wrapper and CuPy execution"""
    
    def __init__(self,
                 device: torch.device,
                 compiled_info: CompiledKernelInfo):
        """
        Initialize TorchCuda executable kernel
        
        Args:
            device: CUDA device to execute on
            compiled_info: Compiled kernel information with C++ wrapper
        """
        self.compiled_info = compiled_info
        
        # Call parent __init__ which will call _initialize_kernel
        super().__init__(
            kernel_type=KernelType.TORCH_CUDA,
            device=device,
        )
        
        # CUDA graph related
        self._graph = None
        self._graph_inputs = None
        self._static_output = None
    
    def _initialize_kernel(self):
        """Initialize kernel-specific components - C++ wrapper with CuPy kernel"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for TORCH_CUDA kernel execution")
            
        # Extract from structured CompiledKernelInfo
        self.compiled_functions = self.compiled_info.compiled_functions
        if not self.compiled_functions:
            raise ValueError("No compiled functions found in CompiledKernelInfo")
        
        # Get the compiled transformed C++ wrapper function
        self.cpp_wrapper = self.compiled_info.cpp_wrapper
        if not self.cpp_wrapper or not self.cpp_wrapper.get("compilation_successful", False):
            raise ValueError("No compiled transformed C++ wrapper found")
        
        self.cpp_extension = self.cpp_wrapper["cpp_extension"]
        
        # Get the C++ function name to target for func_ptr injection
        cpp_function_name = self.cpp_wrapper.get("function_name")
        if not cpp_function_name:
            raise ValueError("No C++ function name found in wrapper info")
        
        # Inject func_ptr parameter only for the specific C++ function
        self.model_new_source = self._inject_func_ptr_parameter(
            self.compiled_info.model_new_source,
            cpp_function_name
        )
        module_to_replace = self._find_module_to_replace(self.model_new_source, cpp_function_name)

        self.namespace = { module_to_replace: self.cpp_extension }
        exec(self.model_new_source, self.namespace)
        
        # Get the first compiled CUDA function and its pointer
        self.func_name = list(self.compiled_functions.keys())[0]
        self.cupy_kernel = self.compiled_functions[self.func_name]
        
        # Get CuPy kernel function pointer for C++ wrapper
        self.func_ptr = int(self.cupy_kernel.kernel.ptr)

        logger.info(f"TorchCudaExecutableKernel initialized with CuPy kernel function pointer: 0x{self.func_ptr:x}")
    
    def _find_module_to_replace(self, source_code: str, cpp_function_name: str) -> str:
        """
        Finds module names that are called with methods in the source code.
        Returns a set of module names that need to be replaced.
        """
        tree = ast.parse(source_code)
        class ModuleFinder(ast.NodeVisitor):
            def __init__(self):
                self.modules_found = []
                
            def visit_Call(self, node):
                # Check if this is a call to our target method
                if isinstance(node.func, ast.Attribute) and node.func.attr == cpp_function_name:
                    # Extract the module it's called on
                    module_name = self._extract_module(node.func.value)
                    if module_name:
                        self.modules_found.append(module_name)
                self.generic_visit(node)
            
            def _extract_module(self, node):
                """Extract module name from different patterns"""
                if isinstance(node, ast.Name):
                    # Pattern: module.method()
                    return node.id
                elif isinstance(node, ast.Attribute):
                    # Pattern: self.module.method() or other.module.method()
                    if isinstance(node.value, ast.Name) and node.value.id == 'self':
                        return node.attr
                    # Could be chained further, extract the last attribute
                    return node.attr
                return None
        
        finder = ModuleFinder()
        finder.visit(tree)
        
        if finder.modules_found:
            # Return the first found (they should all be the same given our assumption)
            return finder.modules_found[0]
        
        raise ValueError(f"No module found calling method '{cpp_function_name}'")

    def _inject_func_ptr_parameter(self, source_code: str, cpp_function_name: str) -> str:
        """
        Adds 'func_ptr: int' parameter to forward method and passes it only to 
        calls matching the specified cpp_function_name.
        
        Args:
            source_code: Original source code string
            cpp_function_name: Name of the C++ function to target (e.g., 'rms_norm_cuda')
        
        Returns:
            Modified source code with func_ptr parameter injected only where needed
        """
        
        class FuncPtrInjector(ast.NodeTransformer):
            def __init__(self, target_function_name):
                self.in_forward = False
                self.target_function_name = target_function_name
                
            def visit_FunctionDef(self, node):
                # Check if this is the forward method
                if node.name == 'forward':
                    self.in_forward = True
                    
                    # Add func_ptr parameter to forward method signature
                    # Create the new parameter with type annotation
                    func_ptr_arg = ast.arg(
                        arg='func_ptr',
                        annotation=ast.Name(id='int', ctx=ast.Load())
                    )
                    
                    # Add it to the function's arguments (before **kwargs if present)
                    node.args.args.append(func_ptr_arg)
                    
                    # Process the body of forward method
                    node = self.generic_visit(node)
                    
                    self.in_forward = False
                else:
                    node = self.generic_visit(node)
                    
                return node
            
            def visit_Call(self, node):
                # Only modify calls inside forward method
                if self.in_forward:
                    # Check if this is an attribute call and matches our target
                    if isinstance(node.func, ast.Attribute):
                        # Check if the attribute name matches our target function
                        if node.func.attr == self.target_function_name:
                            # Add func_ptr as the last argument
                            node.args.append(ast.Name(id='func_ptr', ctx=ast.Load()))
                
                return self.generic_visit(node)
        
        # Parse the source code
        tree = ast.parse(source_code)
        
        # Apply the transformation
        transformer = FuncPtrInjector(cpp_function_name)
        modified_tree = transformer.visit(tree)
        
        # Fix missing locations in the AST
        ast.fix_missing_locations(modified_tree)
        
        # Convert back to source code
        if hasattr(ast, 'unparse'):
            return ast.unparse(modified_tree)
        else:
            # Fallback for Python < 3.9
            try:
                import astor
                return astor.to_source(modified_tree)
            except ImportError:
                raise RuntimeError("Neither ast.unparse (Python 3.9+) nor astor is available for AST to source conversion")

    def _execute_impl(self, *inputs) -> Optional[Any]:
        """Actual execution implementation"""        
        # Critical that torch/cupy device match within same context
        torch.cuda.set_device(self.device)
        with cupy.cuda.Device(self.device.index):
            # Get current func_ptr (in case it changed)
            func_ptr = int(self.cupy_kernel.kernel.ptr)

            return self.model_new.forward(*inputs, func_ptr)
    
    def _set_init_inputs(self, init_inputs) -> None:
        self.model_new = self.namespace['ModelNew'](*init_inputs).cuda(device=self.device)
        assert hasattr(self.model_new, "forward")

class CudaExecutableKernel(BaseExecutableKernel):
    """Raw CUDA kernel executable"""
    
    def _initialize_kernel(self):
        """Initialize raw CUDA kernel"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for CUDA kernel execution")
        
        # Extract raw kernel from compiled info
        self.raw_kernel = self.compiled_info.raw_kernel
        if not self.raw_kernel:
            raise ValueError("No raw kernel found in CompiledKernelInfo")
    
    def _execute_impl(self, *inputs) -> Optional[Any]:
        """Execute raw CUDA kernel"""
        # TODO: Implement raw CUDA kernel execution
        raise NotImplementedError("Raw CUDA kernel execution not yet implemented")


class TorchExecutableKernel(BaseExecutableKernel):
    """Pure PyTorch kernel executable"""
    def __init__(self, device: torch.device, kernel_code: KernelCode):
        self.device = device
        self.kernel_code = kernel_code

        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(self.kernel_code.source_code, {})
        
        # Use fixed seed for deterministic input generation (same as measure_program_time)
        set_seed(42)
        inputs = get_inputs()
        set_seed(42)
        init_inputs = get_init_inputs()
        
        # Move inputs to target device
        self._default_inputs = [
            x.cuda(self.device) if isinstance(x, torch.Tensor) else x 
            for x in inputs
        ]
        self.init_inputs = [
            x.cuda(self.device) if isinstance(x, torch.Tensor) else x 
            for x in init_inputs
        ]
        
        # Create and initialize model
        self.model = Model(*self.init_inputs).cuda(self.device)
        
        # Generate reference output
        with torch.no_grad():
            set_seed(42)
            torch.cuda.synchronize(self.device)
            self.reference_output = self.model(*self._default_inputs)
            torch.cuda.synchronize(self.device)

        # Override default_inputs if IOContract is provided
        if kernel_code.io and kernel_code.io.args:
            try:
                # Generate inputs from IOContract
                generated_inputs = self._generate_inputs_from_io(kernel_code.io)
                if generated_inputs:
                    self._default_inputs = generated_inputs
                    logger.info("Using IOContract-based inputs for TorchExecutableKernel")
            except Exception as e:
                logger.warning(f"Failed to generate inputs from IOContract, using kernelbench defaults: {e}")
        
        # Generate reference output with current inputs
        with torch.no_grad():
            set_seed(42)
            torch.cuda.synchronize(self.device)
            self.reference_output = self.model(*self._default_inputs)
            torch.cuda.synchronize(self.device)    

    def _generate_inputs_from_io(self, io: IOContract) -> List[Any]:
        """Generate inputs from IOContract ArgSpecs"""
        inputs = []
        for arg_spec in io.args:
            if arg_spec.role in ("input", "inout"):  # Only process inputs
                if arg_spec.type == "tensor" and arg_spec.tensor_spec:
                    # Create tensor from spec
                    tensor = materialize_tensor(arg_spec.tensor_spec)
                    # Ensure it's on the right device
                    if tensor.device != self.device:
                        tensor = tensor.to(self.device)
                    inputs.append(tensor)
                elif arg_spec.type in ("int", "float", "bool", "str"):
                    # Use scalar value directly
                    inputs.append(arg_spec.value)
                else:
                    # Unsupported type, skip or raise
                    logger.warning(f"Unsupported ArgSpec type: {arg_spec.type}")
        
        return inputs if inputs else None
    
    def _initialize_kernel(self):
        """Initialize PyTorch model"""
        # TODO: Load PyTorch model from compiled info
        return self._default_inputs, self.init_inputs, self.reference_output, self.model
    
    def _execute_impl(self, *inputs) -> Optional[Any]:
        """Execute PyTorch model"""
        if not inputs:
            return self.model(*self._default_inputs)
        else:
            return self.model(*inputs)
