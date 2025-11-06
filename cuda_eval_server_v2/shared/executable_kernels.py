"""
Executable kernel implementations for different kernel types
Refactored from TorchCudaModelWrapper to use BaseExecutableKernel interface
"""

import logging
import torch
import ast
import sys
import os
from typing import Optional, Any, List, Dict

from shared.models import BaseExecutableKernel, IOContract, KernelCode, KernelType, CompiledKernelInfo
from shared.kernel_metadata import TorchKernelMetadata
from io_contract import IOContractManager

logger = logging.getLogger(__name__)

# Import KernelBench functions
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    if kb_dir not in sys.path:
        sys.path.insert(0, kb_dir)
    
    from KernelBench.eval import load_original_model_and_inputs, set_seed
    KB_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import KernelBench eval functions: {e}")
    KB_FUNCTIONS_AVAILABLE = False
    # Define dummy functions if import fails
    def load_original_model_and_inputs(*args, **kwargs):
        raise ImportError("KernelBench.eval module not available")
    def set_seed(*args, **kwargs):
        pass
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
                 compiled_info: CompiledKernelInfo,
                 io_contract: Optional[IOContract] = None):
        """
        Initialize TorchCuda executable kernel
        
        Args:
            device: CUDA device to execute on
            compiled_info: Compiled kernel information with C++ wrapper
            io_contract: Optional IOContract for input generation
        """
        self.compiled_info = compiled_info
        self.io_contract = io_contract
        
        # Call parent __init__ which will call _initialize_kernel
        super().__init__(
            kernel_type=KernelType.TORCH_CUDA,
            device=device,
            io_contract=io_contract
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
        
        # Initialize inputs - check for get_inputs/get_init_inputs functions first, then IOContract
        self._initialize_inputs()
    
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
    
    def _check_for_input_functions(self, source_code: str) -> bool:
        """Check if get_inputs and get_init_inputs functions exist in the source code"""
        return 'def get_inputs(' in source_code and 'def get_init_inputs(' in source_code
    
    def _initialize_inputs(self):
        """Initialize inputs from source functions or IOContract"""
        # Use IOContract if available
        if self.io_contract and self.io_contract.args:
            try:
                logger.info("Using IOContract for input generation")
                manager = IOContractManager()
                self._default_inputs = manager.generate_inputs(self.io_contract, self.device)
                self.init_inputs = []  # No init inputs from IOContract
                
                # Initialize ModelNew with empty init_inputs
                self._set_init_inputs([])
                
                logger.info(f"Initialized inputs from IOContract: {len(self._default_inputs)} inputs")
                return
            except Exception as e:
                logger.warning(f"Failed to generate inputs from IOContract: {e}")

        # Try to extract from original source code first (KernelBench style)
        original_source = self.compiled_info.original_kernel_source
        if original_source and self._check_for_input_functions(original_source):
            try:
                logger.info("Found get_inputs/get_init_inputs in original source, extracting...")
                # Use the load_original_model_and_inputs function to extract from original source
                Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
                    original_source, 
                    {}  # Use empty namespace for extraction
                )
                
                # Generate inputs using the extracted functions
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
                
                # Initialize ModelNew with init_inputs
                self._set_init_inputs(self.init_inputs)
                
                logger.info(f"Initialized inputs from source functions: {len(self._default_inputs)} inputs")
                return
            except Exception as e:
                logger.warning(f"Failed to extract inputs from source functions: {e}")
        
        # No inputs available - set empty defaults
        self._default_inputs = []
        self.init_inputs = []
               
    
    def _set_init_inputs(self, init_inputs) -> None:
        self.model_new = self.namespace['ModelNew'](*init_inputs).cuda(device=self.device)
        assert hasattr(self.model_new, "forward")

class CudaExecutableKernel(BaseExecutableKernel):
    """CUDA kernel executable with multi-kernel support via RawModule"""
    
    def __init__(self,
                 device: torch.device,
                 raw_module,  # CuPy RawModule - dict-like object with kernels
                 entrypoint_name: str,
                 io_contract: Optional[IOContract] = None):
        """
        Initialize CUDA executable kernel
        
        Args:
            device: CUDA device to execute on
            raw_module: CuPy RawModule (dict-like object with kernels)
            entrypoint_name: Name of the default kernel to execute
            io_contract: IOContract for the entrypoint kernel
        """
        self.raw_module = raw_module
        self.entrypoint_name = entrypoint_name
        self.kernel_name = entrypoint_name  # For compatibility
        
        # Get the entrypoint kernel function
        try:
            self.cupy_kernel = raw_module.get_function(entrypoint_name)
        except Exception as e:
            raise ValueError(f"Failed to get kernel '{entrypoint_name}' from module: {e}")
        
        self.io_contract = io_contract
        
        # Call parent __init__ which will call _initialize_kernel
        super().__init__(
            kernel_type=KernelType.CUDA,
            device=device,
            io_contract=io_contract
        )
    
    def _initialize_kernel(self):
        """Initialize kernel - verify CuPy is available and set up inputs"""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for CUDA kernel execution")
        
        logger.info(f"CudaExecutableKernel initialized with entrypoint: {self.entrypoint_name}")
        
        # Initialize default inputs from IOContract if available
        self._initialize_inputs()
    
    def set_entrypoint(self, kernel_name: str):
        """
        Change the active kernel to execute
        
        Args:
            kernel_name: Name of kernel to set as entrypoint
        """
        try:
            self.cupy_kernel = self.raw_module.get_function(kernel_name)
            self.entrypoint_name = kernel_name
            self.kernel_name = kernel_name  # For compatibility
            logger.info(f"Switched entrypoint to kernel: {kernel_name}")
        except Exception as e:
            raise ValueError(f"Kernel '{kernel_name}' not found in module: {e}")
    
    def execute_kernel(self, kernel_name: str, *inputs):
        """
        Execute a specific kernel by name (temporary switch)
        
        Args:
            kernel_name: Name of kernel to execute
            *inputs: Input arguments for the kernel
            
        Returns:
            Kernel execution results
        """
        prev_entrypoint = self.entrypoint_name
        try:
            self.set_entrypoint(kernel_name)
            return self._execute_impl(*inputs)
        finally:
            self.set_entrypoint(prev_entrypoint)
    
    def _initialize_inputs(self):
        """Initialize default inputs from IOContract"""
        if self.io_contract and self.io_contract.args:
            # Generate inputs from IOContract
            manager = IOContractManager()
            self._default_inputs = manager.generate_inputs(self.io_contract, self.device)
            logger.info("Using IOContract-based inputs for CudaExecutableKernel")
        else:
            logger.info()
            self._default_inputs = []
    
    
    def _get_launch_config(self) -> Dict[str, Any]:
        """Extract launch configuration from IOContract or use defaults"""
        config = {
            "grid": (1, 1, 1),
            "block": (256, 1, 1)
        }
        
        if self.io_contract and self.io_contract.launch:
            launch = self.io_contract.launch
            
            # Grid dimensions
            if launch.grid:
                config["grid"] = (launch.grid.x, launch.grid.y, launch.grid.z)
            
            # Block dimensions
            if launch.block:
                config["block"] = (launch.block.x, launch.block.y, launch.block.z)
        
        return config
    
    def _execute_impl(self, *inputs) -> Optional[Any]:
        """
        Execute raw CUDA kernel
        
        Args:
            *inputs: Input tensors and scalars
            
        Returns:
            Output tensors or None
        """
        # Use default inputs if none provided
        if not inputs:
            inputs = self._default_inputs
        
        if not inputs:
            raise ValueError("No inputs provided for CUDA kernel execution")
        
        # Set CUDA device
        torch.cuda.set_device(self.device)
        
        # Get PyTorch's current CUDA stream
        torch_stream = torch.cuda.current_stream(self.device)
        
        with cupy.cuda.Device(self.device.index):
            # Use PyTorch's CUDA stream for CuPy operations
            with cupy.cuda.ExternalStream(torch_stream.cuda_stream):
                # Convert torch tensors to CuPy arrays, preserve scalars
                cupy_args = []
                output_indices = []
                
                for i, arg in enumerate(inputs):
                    if torch.is_tensor(arg):
                        # Ensure tensor is contiguous
                        if not arg.is_contiguous():
                            arg = arg.contiguous()
                        
                        cupy_args.append(cupy.asarray(arg))
                        
                        # Track output indices based on IOContract
                        if self.io_contract and i < len(self.io_contract.args):
                            arg_spec = self.io_contract.args[i]
                            if arg_spec.role in ("output", "inout"):
                                output_indices.append(i)
                    else:
                        # Scalar argument
                        cupy_args.append(arg)
                
                # Get launch configuration
                launch_config = self._get_launch_config()
                
                # Execute kernel (will run on PyTorch's stream)
                self.cupy_kernel(
                    launch_config["grid"],
                    launch_config["block"],
                    tuple(cupy_args)
                )
                
                # Convert outputs back to torch tensors
                # Since we used shared memory via CUDA array interface, the original torch tensors
                # already have the updated values, so we can just return them
                if output_indices:
                    outputs = []
                    for idx in output_indices:
                        if idx < len(inputs):
                            # Return the original torch tensor (which now has updated values)
                            if torch.is_tensor(inputs[idx]):
                                outputs.append(inputs[idx])
                    
                    if len(outputs) == 1:
                        return outputs[0]
                    else:
                        return tuple(outputs)
                else:
                    # No explicit outputs marked, return all tensor arguments
                    outputs = []
                    for arg in inputs:
                        if torch.is_tensor(arg):
                            outputs.append(arg)
                    
                    if outputs:
                        return tuple(outputs) if len(outputs) > 1 else outputs[0]
                    else:
                        return None
            


class TorchExecutableKernel(BaseExecutableKernel):
    """Pure PyTorch kernel executable with optional function targeting"""
    def __init__(self, device: torch.device, kernel_code: KernelCode):
        self.device = device
        self.kernel_code = kernel_code
        self.io_contract = None
        
        # Check if we should use targeted execution (requires both IOContract and metadata)
        if kernel_code.io and kernel_code.io.args and kernel_code.metadata:
            self.io_contract = kernel_code.io
            self._use_targeted_execution = True
            self._setup_targeted_execution()
        else:
            # Fall back to KernelBench pattern
            self._use_targeted_execution = False
            self._setup_kernelbench_execution()
    
    def _setup_targeted_execution(self):
        """Set up targeted function/method execution based on metadata"""
        try:
            # Compile and execute source code
            self.namespace = {}
            compiled_code = compile(self.kernel_code.source_code, "<string>", "exec")
            exec(compiled_code, self.namespace)
            
            # Get typed metadata
            metadata = self.kernel_code.get_typed_metadata()
            if not isinstance(metadata, TorchKernelMetadata):
                # Create from dict if needed
                metadata = TorchKernelMetadata.from_dict(
                    self.kernel_code.metadata if isinstance(self.kernel_code.metadata, dict) else {}
                )
            
            # Resolve execution target
            if metadata.function_name:
                # Standalone function
                if metadata.function_name not in self.namespace:
                    raise KeyError(f"Function '{metadata.function_name}' not found in source code")
                self.target_callable = self.namespace[metadata.function_name]
                self.is_method = False
                logger.info(f"TorchExecutableKernel targeting standalone function: {metadata.function_name}")
            else:
                # Class method
                class_name = metadata.class_name or "Model"
                method_name = metadata.method_name or "forward"
                
                if class_name not in self.namespace:
                    raise KeyError(f"Class '{class_name}' not found in source code")
                
                cls = self.namespace[class_name]
                
                # Check if we need init inputs
                init_inputs = self._get_init_inputs_from_io()
                
                # Create instance
                self.instance = cls(*init_inputs)
                if hasattr(self.instance, 'cuda'):
                    self.instance = self.instance.cuda(self.device)
                elif hasattr(self.instance, 'to'):
                    self.instance = self.instance.to(self.device)
                
                # Get the method
                if not hasattr(self.instance, method_name):
                    raise AttributeError(f"Method '{method_name}' not found in class '{class_name}'")
                
                self.target_callable = getattr(self.instance, method_name)
                self.is_method = True
                logger.info(f"TorchExecutableKernel targeting {class_name}.{method_name}")
            
            # Generate inputs from IOContract
            manager = IOContractManager()
            self._default_inputs = manager.generate_inputs(self.kernel_code.io, self.device)
            
            # Generate reference output if possible
            self._generate_reference_output()
            
        except Exception as e:
            logger.error(f"Failed to set up targeted execution: {e}")
            logger.info("Falling back to KernelBench execution pattern")
            self._use_targeted_execution = False
            self._setup_kernelbench_execution()
    
    def _setup_kernelbench_execution(self):
        """Set up traditional KernelBench execution pattern"""
        # Check if IOContract is available first
        if self.kernel_code.io and self.kernel_code.io.args:
            try:
                # Generate inputs from IOContract
                manager = IOContractManager()
                generated_inputs = manager.generate_inputs(self.kernel_code.io, self.device)
                if generated_inputs:
                    self._default_inputs = generated_inputs
                    logger.info("Using IOContract-based inputs for TorchExecutableKernel")
                    # Try to create a simple model wrapper
                    self._create_simple_model_wrapper()
                    return
            except Exception as e:
                logger.warning(f"Failed to generate inputs from IOContract, using kernelbench defaults: {e}")
        
        # Fallback to traditional KernelBench pattern
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(self.kernel_code.source_code, {})
        
        # Use fixed seed for deterministic input generation
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
        
        # Generate reference output with current inputs
        with torch.no_grad():
            set_seed(42)
            torch.cuda.synchronize(self.device)
            self.reference_output = self.model(*self._default_inputs)
            torch.cuda.synchronize(self.device)
    
    def _create_simple_model_wrapper(self):
        """Create a simple model wrapper when only IOContract is available"""
        try:
            # Execute the source to get Model class
            namespace = {}
            exec(compile(self.kernel_code.source_code, "<string>", "exec"), namespace)
            
            if "Model" in namespace:
                Model = namespace["Model"]
                # Try to instantiate with no args or empty args
                try:
                    self.model = Model().cuda(self.device)
                except:
                    # Try with empty init inputs
                    self.model = Model(*[]).cuda(self.device)
                
                # Generate reference output
                with torch.no_grad():
                    set_seed(42)
                    torch.cuda.synchronize(self.device)
                    self.reference_output = self.model(*self._default_inputs)
                    torch.cuda.synchronize(self.device)
            else:
                # No Model class found, store reference as None
                self.model = None
                self.reference_output = None
        except Exception as e:
            logger.warning(f"Could not create model wrapper: {e}")
            self.model = None
            self.reference_output = None
    
    def _get_init_inputs_from_io(self) -> List[Any]:
        """Extract init inputs from IOContract if available"""
        # For now, return empty list - could be extended to support init args in IOContract
        return []
    
    def _generate_reference_output(self):
        """Generate reference output for targeted execution"""
        try:
            with torch.no_grad():
                set_seed(42)
                torch.cuda.synchronize(self.device)
                self.reference_output = self.target_callable(*self._default_inputs)
                torch.cuda.synchronize(self.device)
        except Exception as e:
            logger.warning(f"Could not generate reference output: {e}")
            self.reference_output = None    

    
    def _initialize_kernel(self):
        """Initialize PyTorch model"""
        # TODO: Load PyTorch model from compiled info
        return self._default_inputs, self.init_inputs, self.reference_output, self.model
    
    def _execute_impl(self, *inputs) -> Optional[Any]:
        """Execute PyTorch kernel based on execution mode"""
        if self._use_targeted_execution:
            # Use targeted execution
            execution_inputs = inputs if inputs else self._default_inputs
            return self.target_callable(*execution_inputs)
        else:
            # Use traditional model execution
            if not inputs:
                return self.model(*self._default_inputs) if self.model else None
            else:
                return self.model(*inputs) if self.model else None


class MultiKernelExecutableKernel(BaseExecutableKernel):
    """
    Executable multi-kernel sequence that:
      - loads Python source with mixed kernel types (Triton, CUDA, Torch)
      - executes entry point function
      - returns outputs
    """

    def __init__(
        self,
        kernel_code: KernelCode,
        device: torch.device,
        metadata: Any,  # MultiKernelMetadata or dict
    ):
        from shared.kernel_metadata import MultiKernelMetadata

        assert kernel_code.kernel_type == KernelType.MULTI_KERNEL, "MultiKernelExecutableKernel requires MULTI_KERNEL type"

        self.kernel_code = kernel_code
        self.device = device

        # Extract metadata
        if isinstance(metadata, MultiKernelMetadata):
            self.entry_point_name = metadata.entry_point
        elif isinstance(metadata, dict):
            self.entry_point_name = metadata["entry_point"]
        else:
            raise ValueError("MultiKernelMetadata required with 'entry_point' field")

        self._loader_info: Optional[tuple] = None  # (temp_dir, module)
        self._entry_point_func = None
        self.io_contract = kernel_code.io

        # Call parent __init__ which will call _initialize_kernel
        super().__init__(
            kernel_type=kernel_code.kernel_type,
            device=device,
            io_contract=kernel_code.io
        )

    def _initialize_kernel(self):
        """Load module and extract entry point function"""
        import tempfile
        import pathlib
        import importlib.util
        import sys
        import hashlib

        # Create temp file
        temp_dir = tempfile.mkdtemp(prefix="multi_kernel_")
        temp_path = pathlib.Path(temp_dir) / "user_module.py"
        temp_path.write_text(self.kernel_code.source_code, encoding="utf-8")

        # Load as module
        source_hash = hashlib.md5(self.kernel_code.source_code.encode()).hexdigest()[:8]
        module_name = f"user_multi_kernel_{source_hash}"

        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for {temp_path}")

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

        self._loader_info = (temp_dir, mod)

        # Extract entry point function
        if not hasattr(mod, self.entry_point_name):
            available = [k for k in dir(mod) if not k.startswith('_')]
            raise RuntimeError(
                f"Entry point '{self.entry_point_name}' not found in source. "
                f"Available: {available}"
            )

        self._entry_point_func = getattr(mod, self.entry_point_name)
        logger.info(f"MultiKernelExecutableKernel initialized with entry point: {self.entry_point_name}")

        # Initialize inputs from IOContract
        self._initialize_inputs()

    def _initialize_inputs(self):
        """Generate default inputs from IOContract"""
        if self.io_contract and self.io_contract.args:
            manager = IOContractManager()
            self._default_inputs = manager.generate_inputs(self.io_contract, self.device)
            logger.info(f"Initialized {len(self._default_inputs)} default inputs from IOContract")
        else:
            logger.warning("No IOContract provided - execution will require explicit inputs")
            self._default_inputs = []

    def _execute_impl(self, *inputs) -> Optional[Any]:
        """
        Execute entry point function

        Args:
            *inputs: Input tensors/scalars (uses _default_inputs if not provided)

        Returns:
            Output from entry point function
        """
        # Use default inputs if none provided
        if not inputs:
            inputs = self._default_inputs

        if not inputs:
            raise ValueError("No inputs provided and no default inputs available")

        # Set device
        torch.cuda.set_device(self.device)

        # Call entry point function
        return self._entry_point_func(*inputs)

    def cleanup(self):
        """Cleanup temp files"""
        if self._loader_info:
            temp_dir, _ = self._loader_info
            import pathlib
            if pathlib.Path(temp_dir).exists():
                import shutil
                shutil.rmtree(temp_dir)
            self._loader_info = None

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass
