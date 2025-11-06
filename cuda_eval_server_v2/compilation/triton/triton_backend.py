"""
Triton compilation strategy implementation
"""

import logging
import torch
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import importlib.util
import tempfile
import pathlib
import sys

from compilation.base_compiler import BaseCompilationBackend
from shared.models import BaseExecutableKernel, KernelCode, IOContract, KernelType
from shared.kernel_metadata import TritonKernelMetadata
from io_contract import IOContractManager

logger = logging.getLogger(__name__)


class _TritonModuleLoader:
    """Simple module loader for Triton source code"""
    
    def __init__(self, source: str):
        self.source = source
        self.ns: Dict[str, Any] = {}
        self._temp_dir = None
        self._temp_path = None
        
    def exec(self):
        """Execute the source code and populate namespace"""
        self._temp_dir = tempfile.mkdtemp(prefix="triton_kernel_")
        self._temp_path = pathlib.Path(self._temp_dir) / "user_triton_module.py"
        self._temp_path.write_text(self.source, encoding="utf-8")

        import hashlib
        source_hash = hashlib.md5(self.source.encode()).hexdigest()[:8]
        module_name = f"user_triton_module_{source_hash}"
        
        spec = importlib.util.spec_from_file_location(module_name, self._temp_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        
        self.ns = mod.__dict__
        return self
    
    def cleanup(self):
        """Clean up temporary files"""
        if self._temp_dir and pathlib.Path(self._temp_dir).exists():
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self._temp_path = None
    
    def __del__(self):
        try:
            self.cleanup()
        except:
            pass

    def list_kernels(self) -> List[Tuple[str, Any]]:
        """List all JIT kernels in the module"""
        out: List[Tuple[str, Any]] = []
        
        # Try to identify Triton JIT functions
        try:
            import triton.runtime.jit as trjit
            JIT = trjit.JITFunction
        except Exception:
            JIT = None

        for k, v in self.ns.items():
            if JIT is not None and isinstance(v, JIT):
                out.append((k, v))
                continue
            # Fallback check for JIT-like objects
            if hasattr(v, "__getitem__") and hasattr(v, "__call__") and hasattr(v, "fn"):
                out.append((k, v))
        
        return out


class TritonExecutableKernel(BaseExecutableKernel):
    """
    Executable Triton kernel that:
      - loads user source code,
      - generates grid, inputs args from iocontract
      - can do correctness checks and profiling (CUDA Events & NCU).
    """
    
    def __init__(
        self,
        kernel_code: KernelCode,
        device: torch.device,
        kernel_name: Optional[str] = None,
        metadata: Optional[Union[TritonKernelMetadata, Dict[str, Any]]] = None,
    ):
        assert kernel_code.kernel_type == KernelType.TRITON, "TritonExecutableKernel requires KernelType.TRITON"
        
        self.kernel_code = kernel_code
        
        # Extract kernel_name from metadata if provided, otherwise use parameter
        if metadata:
            if isinstance(metadata, TritonKernelMetadata):
                self.kernel_name = metadata.kernel_name or kernel_name
            elif isinstance(metadata, dict):
                self.kernel_name = metadata.get("kernel_name") or kernel_name
            else:
                self.kernel_name = kernel_name
        else:
            self.kernel_name = kernel_name
            
        self._loader: Optional[_TritonModuleLoader] = None
        self._jit_func = None
        self.io_contract = kernel_code.io  # Set io_contract before parent init
        
        # Call parent __init__ which will call _initialize_kernel
        super().__init__(
            kernel_type=kernel_code.kernel_type,
            device=device,
            io_contract=kernel_code.io
        )

    def _initialize_kernel(self):
        """Initialize kernel - load module and set up default inputs"""
        # Load Triton module
        self._loader = _TritonModuleLoader(self.kernel_code.source_code).exec()
        
        # Get available kernels
        kernels = self._loader.list_kernels()
        
        if not kernels:
            raise RuntimeError("No @triton.jit kernels found in source.")
        
        # Select kernel by name or use first
        if self.kernel_name:
            for k, v in kernels:
                if k == self.kernel_name:
                    self._jit_func = v
                    break
            if self._jit_func is None:
                raise RuntimeError(f"Kernel '{self.kernel_name}' not found. Available: {[k for k,_ in kernels]}")
        else:
            self.kernel_name, self._jit_func = kernels[0]
            logger.info(f"Using first kernel found: {self.kernel_name}")
        
        # Initialize default inputs from IOContract (like CudaExecutableKernel does)
        if self.io_contract and self.io_contract.args:
            manager = IOContractManager()
            # Get ALL inputs including meta parameters
            self._default_inputs = manager.generate_inputs(self.io_contract, self.device)
            logger.info(f"Initialized {len(self._default_inputs)} default inputs from IOContract")
        else:
            logger.warning("No IOContract provided - kernel execution will require explicit inputs")
            self._default_inputs = []


    def _get_grid_from_io(self) -> Tuple[int, ...]:
        """Extract grid configuration from IOContract or use default"""
        if self.io_contract and self.io_contract.launch and self.io_contract.launch.grid:
            grid = self.io_contract.launch.grid
            return (grid.x, grid.y, grid.z)
        else:
            # Default grid
            return (1,)

    def _get_meta_kwargs(self) -> Dict[str, Any]:
        """Extract meta parameters (constexpr and launch config) from IOContract"""
        meta = {}
        
        if not self.io_contract:
            return meta
        
        # Get constexpr/meta parameters
        for arg_spec in self.io_contract.args:
            if arg_spec.is_meta and arg_spec.value is not None:
                # Convert bool to int for Triton
                value = int(arg_spec.value) if isinstance(arg_spec.value, bool) else arg_spec.value
                meta[arg_spec.name] = value
        
        # Add num_warps/num_stages from launch config
        if self.io_contract.launch:
            if self.io_contract.launch.num_warps is not None:
                meta["num_warps"] = int(self.io_contract.launch.num_warps)
            if self.io_contract.launch.num_stages is not None:
                meta["num_stages"] = int(self.io_contract.launch.num_stages)
        
        return meta

    def _identify_output_indices(self) -> List[int]:
        """Identify which positional arguments are outputs based on IOContract"""
        output_indices = []
        
        if not self.io_contract:
            return output_indices
        
        # Track position among non-meta args
        pos_idx = 0
        for arg_spec in self.io_contract.args:
            if not arg_spec.is_meta:
                if arg_spec.role in ("output", "inout"):
                    output_indices.append(pos_idx)
                pos_idx += 1
        
        return output_indices

    def _execute_impl(self, *inputs) -> Optional[Any]:
        """
        Execute Triton kernel with IOContract-based configuration
        
        Args:
            *inputs: Input tensors and scalars (optional, uses _default_inputs if not provided)
            
        Returns:
            Output tensors based on IOContract role specifications
        """
        # Use default inputs if none provided
        if not inputs:
            inputs = list(self._default_inputs)
        
        if not inputs and not self.io_contract:
            raise ValueError("No inputs provided and no IOContract available for execution")
        
        # Get grid configuration
        grid = self._get_grid_from_io()
        
        # Get meta parameters (constexpr + num_warps/stages)
        # meta_kwargs = self._get_meta_kwargs()
        
        # Execute kernel with grid
        self._jit_func[grid](*inputs)
        
        # Return outputs based on IOContract roles
        output_indices = self._identify_output_indices()
        
        if output_indices:
            outputs = [inputs[i] for i in output_indices]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)
        else:
            # No explicit outputs marked - return all tensors as fallback
            tensor_outputs = [x for x in inputs if torch.is_tensor(x)]
            if tensor_outputs:
                return tensor_outputs[0] if len(tensor_outputs) == 1 else tuple(tensor_outputs)
            else:
                return None

    def cleanup(self):
        """Cleanup resources"""
        if self._loader:
            self._loader.cleanup()
            self._loader = None
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass


class TritonCompilationBackend(BaseCompilationBackend):
    """Simplified compilation backend for Triton kernels"""
    
    def __init__(self, cache_dir: str = "/tmp/triton_cache"):
        self.cache_dir = cache_dir
        self.logger = logger
        
    def compile(self, kernel: KernelCode, gpu_id: int, **kwargs) -> BaseExecutableKernel:
        """
        Compile Triton kernel into executable form
        
        Args:
            kernel: KernelCode with Triton source and IOContract
            gpu_id: GPU device ID
            
        Returns:
            TritonExecutableKernel ready for execution
        """
        device = torch.device(f"cuda:{gpu_id}")
        
        # Validate kernel type
        if kernel.kernel_type != KernelType.TRITON:
            raise ValueError(f"Expected TRITON kernel type, got {kernel.kernel_type}")
        
        # Require IOContract for simplified flow
        if not kernel.io:
            raise ValueError(
                "IOContract is required for Triton kernel compilation. "
                "Please provide input/output specifications via IOContract."
            )
        
        # Extract metadata and convert if necessary
        metadata = kernel.get_typed_metadata()
        self.logger.info(f"Raw metadata from kernel: {metadata}, type: {type(metadata)}")
        
        if metadata and not isinstance(metadata, (TritonKernelMetadata, dict)):
            # Try to convert to dict if it's another type
            metadata = metadata if isinstance(metadata, dict) else None
        
        # Convert dict to TritonKernelMetadata for consistency
        if isinstance(metadata, dict):
            self.logger.info(f"Converting dict metadata to TritonKernelMetadata: {metadata}")
            metadata = TritonKernelMetadata.from_dict(metadata)
        
        # Extract basic kernel information (optional, for logging)
        self.logger.info("Extracting Triton kernel information")
        
        # Determine kernel name priority: metadata > extracted > None
        kernel_name = None
        if metadata and hasattr(metadata, 'kernel_name') and metadata.kernel_name:
            kernel_name = metadata.kernel_name
            self.logger.info(f"Using kernel name from metadata: {kernel_name}")
        else:
            self.logger.warning(f"Could not extract kernel info from metadata: {metadata}, will use first kernel found")
        
        # Build executable kernel with IOContract and metadata
        return TritonExecutableKernel(
            kernel_code=kernel,
            device=device,
            kernel_name=kernel_name,
            metadata=metadata
        )
    