"""
MCP Server for CUDA Evaluation Server V2 (FastMCP-based)
Provides Model Context Protocol interface for Claude to interact with GPU kernel evaluation
"""

import logging
from typing import Any, Dict, Optional, Literal
from mcp.server import FastMCP

# Server imports
from orchestration.job_manager import JobManager
from shared.models import (
    EvaluationRequest, CompareRequest,
    KernelCode, KernelType, IOContract
)

# Set up logging
logger = logging.getLogger(__name__)


def create_mcp_server(job_manager: JobManager) -> FastMCP:
    """
    Factory function to create FastMCP server with JobManager

    Args:
        job_manager: JobManager instance for job orchestration

    Returns:
        Configured FastMCP server instance
    """
    mcp = FastMCP(
        name="cuda-eval-server",
        streamable_http_path="/"
    )

    # Store job_manager reference for tool access
    # We'll use closure to capture it

    @mcp.tool()
    async def evaluate_kernel(
        kernel_source: str,
        kernel_type: Literal["torch", "torch_cuda", "triton", "cuda", "multi_kernel"],
        reference_kernel_source: Optional[str] = None,
        reference_kernel_type: Optional[str] = None,
        io_contract: Optional[Dict[str, Any]] = None,
        num_trials: int = 100,
        num_warmup: int = 10,
        enable_device_metrics: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        timeout: int = 120,
        metadata: Optional[Dict[str, Any]] = None,
        reference_kernel_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compile, validate, and profile a GPU kernel with subprocess isolation for safety.

        ## SUPPORTED KERNEL TYPES

        - **TORCH**: Pure PyTorch models (IOContract optional, auto-generated from model)
        - **TORCH_CUDA**: PyTorch with embedded CUDA via load_inline (IOContract optional)
        - **TRITON**: Triton JIT kernels (IOContract REQUIRED with meta parameters)
        - **CUDA**: Raw CUDA C++ (IOContract REQUIRED with launch config)
        - **MULTI_KERNEL**: Python scripts mixing kernel types (IOContract REQUIRED + entry_point in metadata)

        ## ERROR HANDLING PHILOSOPHY

        **IMPORTANT**: Kernel failures return SUCCESS (HTTP 200) with error flags:
        - Compilation failure: `compiled=false`, `compilation_error="..."`
        - Validation failure: `correctness=false`, `validation_error="..."`
        - Always check these flags to determine if the kernel worked correctly
        - Server errors (HTTP 500) only occur for infrastructure failures

        ## SOURCE CODE CONSTRAINTS

        **All Kernels**:
        - Must be self-contained with all imports
        - No execution code in global scope
        - No `if __name__ == '__main__'` blocks
        - Source is executed via exec() - ensure proper escaping

        **TORCH Specific**:
        - Define Model class or use metadata to target functions
        - No mixed kernel types (no Triton/CUDA code)

        **TORCH_CUDA Specific**:
        - Must follow exact load_inline pattern
        - Define cuda_source and cpp_source as variables
        - Must have ModelNew class (not Model)

        **TRITON Specific**:
        - Must have @triton.jit decorated kernel
        - Meta parameters (constexpr) marked with is_meta=true
        - IOContract MANDATORY with grid configuration

        **MULTI_KERNEL Specific**:
        - MUST specify entry_point in metadata (function to call)
        - Entry point function must exist in source
        - Can mix Triton, CUDA, and PyTorch in single script

        ## SAFETY FEATURES

        - **Subprocess Isolation**: Kernel crashes don't affect server
        - **GPU Resource Management**: Automatic allocation and cleanup
        - **Memory Reset**: GPU state cleared between evaluations

        ## EXAMPLE - Triton kernel

        ```json
        {
          "kernel_source": "import triton\\nimport triton.language as tl\\n@triton.jit\\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):\\n    pid = tl.program_id(0)\\n    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\\n    mask = offs < n\\n    x = tl.load(x_ptr + offs, mask=mask)\\n    y = tl.load(y_ptr + offs, mask=mask)\\n    tl.store(out_ptr + offs, x + y, mask=mask)",
          "kernel_type": "triton",
          "io_contract": {
            "args": [
              {"name": "x_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "randn", "seed": 42}}},
              {"name": "y_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "ones"}}},
              {"name": "out_ptr", "type": "tensor", "role": "output", "tensor_spec": {"shape": [1024], "dtype": "float32"}},
              {"name": "n", "type": "int", "value": 1024, "role": "input"},
              {"name": "BLOCK_SIZE", "type": "int", "value": 256, "role": "input", "is_meta": true}
            ],
            "launch": {"grid": {"x": 4}, "num_warps": 4}
          },
          "num_trials": 100
        }
        ```

        ## EXAMPLE - Torch kernel (no IOContract needed)

        ```json
        {
          "kernel_source": "import torch\\n\\nclass Model(torch.nn.Module):\\n    def forward(self, x):\\n        return x * 2\\n\\ndef get_inputs():\\n    return [torch.randn(1024)]",
          "kernel_type": "torch"
        }
        ```

        ## EXAMPLE - Multi-kernel with entry_point

        ```json
        {
          "kernel_source": "import torch\\nimport triton\\nimport triton.language as tl\\n\\n@triton.jit\\ndef add_kernel(...):\\n    pass\\n\\ndef run(x, y):\\n    # Mix Triton and PyTorch\\n    output = torch.empty_like(x)\\n    add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)\\n    return output * 2",
          "kernel_type": "multi_kernel",
          "metadata": {"entry_point": "run"},
          "io_contract": {"args": [...]}
        }
        ```

        Args:
            kernel_source: The source code of the kernel to evaluate
            kernel_type: Type of kernel (torch, torch_cuda, triton, cuda, multi_kernel)
            reference_kernel_source: Optional reference kernel source for correctness validation and performance comparison
            reference_kernel_type: Type of reference kernel (defaults to 'torch' if not specified)
            io_contract: Input/output contract specification. REQUIRED for TRITON, CUDA, and MULTI_KERNEL types.
                        Optional for TORCH and TORCH_CUDA (auto-generated from get_inputs() if not provided).
                        Defines kernel arguments, tensor specifications, and launch configuration.
            num_trials: Number of profiling trials (default: 100)
            num_warmup: Number of warmup runs before profiling (default: 10)
            enable_device_metrics: Enable NCU profiling for detailed GPU metrics (default: false)
            rtol: Relative tolerance for validation (default: 1e-5)
            atol: Absolute tolerance for validation (default: 1e-8)
            timeout: Maximum time to wait for job completion in seconds (default: 120)
            metadata: Kernel metadata for advanced targeting and configuration.
                     Contents depend on kernel_type:
                     - TORCH: {"function_name": "...", "class_name": "...", "method_name": "..."} for targeting specific functions/methods
                     - TRITON: {"kernel_name": "..."} for selecting specific kernel when multiple @triton.jit exist
                     - MULTI_KERNEL: {"entry_point": "..."} REQUIRED - specifies which function to call
                     - CUDA: {"kernel_name": "...", "compile_options": [...]}
            reference_kernel_metadata: Metadata for reference kernel (same structure as metadata)

        Returns:
            Evaluation results including compilation status, validation, and profiling metrics
        """
        # Validate IOContract if provided
        is_valid, error_msg = _validate_io_contract(io_contract, kernel_type)

        if not is_valid:
            return {
                "status": "failed",
                "error": f"IOContract validation failed: {error_msg}",
                "compiled": False,
                "validation_error": error_msg
            }

        # Build kernel object
        kernel = KernelCode(
            source_code=kernel_source,
            kernel_type=KernelType(kernel_type),
            io=_parse_io_contract(io_contract) if io_contract else None,
            metadata=metadata or {}
        )

        # Build reference kernel if provided (comparison mode)
        ref_kernel = None
        if reference_kernel_source:
            ref_kernel = KernelCode(
                source_code=reference_kernel_source,
                kernel_type=KernelType(reference_kernel_type or "torch"),
                io=None,  # Reference kernels typically don't need IO contract
                metadata=reference_kernel_metadata or {}
            )

        # Create appropriate request based on mode
        if ref_kernel:
            # Comparison mode
            request = CompareRequest(
                custom_kernel=kernel,
                ref_kernel=ref_kernel,
                num_trials=num_trials,
                timeout=timeout,
                atol=atol,
                rtol=rtol
            )
        else:
            # Single kernel evaluation mode
            request = EvaluationRequest(
                kernel=kernel,
                num_trials=num_trials,
                timeout=timeout
            )

        # Submit job
        job_id = await job_manager.submit_evaluation_job(request)
        logger.info(f"Submitted evaluation job: {job_id}")

        # Wait for completion
        result = await job_manager.wait_for_completion(job_id, timeout=timeout)

        if result is None:
            # Job failed or timed out
            job_state = await job_manager.get_job_status(job_id)
            if job_state and job_state.status == "failed":
                error_msg = job_state.error or "Job failed with unknown error"
            else:
                error_msg = f"Job timed out after {timeout} seconds"

            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }

        # Convert result to dict
        result_dict = _convert_result_to_dict(result)

        return result_dict

    @mcp.tool()
    async def compare_kernels(
        ref_kernel_source: str,
        ref_kernel_type: Literal["torch", "torch_cuda", "triton", "cuda", "multi_kernel"],
        custom_kernel_source: str,
        custom_kernel_type: Literal["torch", "torch_cuda", "triton", "cuda", "multi_kernel"],
        ref_kernel_io: Optional[Dict[str, Any]] = None,
        custom_kernel_io: Optional[Dict[str, Any]] = None,
        num_trials: int = 100,
        num_warmup: int = 10,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        timeout: int = 120,
        ref_kernel_metadata: Optional[Dict[str, Any]] = None,
        custom_kernel_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare the performance and correctness of two GPU kernels.

        This tool compiles, validates, and profiles both a reference kernel and a custom kernel,
        then compares their outputs for correctness and their execution times for performance.

        ## SUPPORTED KERNEL TYPES

        - **TORCH**: Pure PyTorch models (IOContract optional, auto-generated from get_inputs())
        - **TORCH_CUDA**: PyTorch with embedded CUDA (IOContract optional)
        - **TRITON**: Triton kernels (IOContract REQUIRED)
        - **CUDA**: Raw CUDA C++ (IOContract REQUIRED)
        - **MULTI_KERNEL**: Mixed kernel types (IOContract REQUIRED + entry_point in metadata)

        ## RETURNS

        - Compilation status for both kernels
        - Correctness validation (outputs compared with tolerance)
        - Performance comparison (runtime statistics for both kernels)
        - Speedup factor (custom vs reference)

        ## IMPORTANT NOTES

        - Kernel failures return success (HTTP 200) with error flags (compiled=false, correctness=false)
        - Check response fields to determine if kernels compiled and validated successfully
        - Both kernels receive identical inputs for fair comparison
        - Subprocess isolation prevents kernel crashes from affecting the server

        ## EXAMPLE - Comparing PyTorch reference with Triton optimization

        ```json
        {
          "ref_kernel_source": "import torch\\nclass Model(torch.nn.Module):\\n    def forward(self, x, y):\\n        return x + y",
          "ref_kernel_type": "torch",
          "custom_kernel_source": "import triton\\nimport triton.language as tl\\n@triton.jit\\ndef add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):\\n    pid = tl.program_id(0)\\n    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\\n    mask = offs < n\\n    x = tl.load(x_ptr + offs, mask=mask)\\n    y = tl.load(y_ptr + offs, mask=mask)\\n    tl.store(out_ptr + offs, x + y, mask=mask)",
          "custom_kernel_type": "triton",
          "custom_kernel_io": {
            "args": [
              {"name": "x_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "randn", "seed": 42}}},
              {"name": "y_ptr", "type": "tensor", "role": "input", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "ones"}}},
              {"name": "out_ptr", "type": "tensor", "role": "output", "tensor_spec": {"shape": [1024], "dtype": "float32"}},
              {"name": "n", "type": "int", "value": 1024, "role": "input"},
              {"name": "BLOCK_SIZE", "type": "int", "value": 256, "role": "input", "is_meta": true}
            ],
            "launch": {"grid": {"x": 4}, "num_warps": 4}
          },
          "num_trials": 100,
          "atol": 1e-5,
          "rtol": 1e-5
        }
        ```

        Args:
            ref_kernel_source: Source code of the reference kernel (usually PyTorch)
            ref_kernel_type: Type of reference kernel (defaults to 'torch')
            custom_kernel_source: Source code of the custom kernel to compare
            custom_kernel_type: Type of custom kernel
            ref_kernel_io: IOContract for reference kernel (optional for TORCH/TORCH_CUDA)
            custom_kernel_io: IOContract for custom kernel (REQUIRED for TRITON/CUDA/MULTI_KERNEL)
            num_trials: Number of profiling trials (default: 100)
            num_warmup: Number of warmup runs before profiling (default: 10)
            atol: Absolute tolerance for correctness validation (default: 1e-5)
            rtol: Relative tolerance for correctness validation (default: 1e-5)
            timeout: Maximum time to wait for job completion in seconds (default: 120)
            ref_kernel_metadata: Reference kernel metadata for advanced targeting:
                                - TORCH: {"function_name": "...", "class_name": "...", "method_name": "..."}
                                - TRITON: {"kernel_name": "..."}
                                - MULTI_KERNEL: {"entry_point": "..."} REQUIRED
                                - CUDA: {"kernel_name": "...", "compile_options": [...]}
            custom_kernel_metadata: Custom kernel metadata for advanced targeting (same structure as ref_kernel_metadata)

        Returns:
            Comparison results with correctness and performance metrics including speedup
        """
        # Validate IOContracts if provided
        is_valid, error_msg = _validate_io_contract(ref_kernel_io, ref_kernel_type)

        if not is_valid:
            return {
                "status": "failed",
                "error": f"Reference kernel IOContract validation failed: {error_msg}",
                "compiled": False,
                "validation_error": error_msg
            }

        is_valid, error_msg = _validate_io_contract(custom_kernel_io, custom_kernel_type)

        if not is_valid:
            return {
                "status": "failed",
                "error": f"Custom kernel IOContract validation failed: {error_msg}",
                "compiled": False,
                "validation_error": error_msg
            }

        # Build reference kernel
        ref_kernel = KernelCode(
            source_code=ref_kernel_source,
            kernel_type=KernelType(ref_kernel_type),
            io=_parse_io_contract(ref_kernel_io) if ref_kernel_io else None,
            metadata=ref_kernel_metadata or {}
        )

        # Build custom kernel
        custom_kernel = KernelCode(
            source_code=custom_kernel_source,
            kernel_type=KernelType(custom_kernel_type),
            io=_parse_io_contract(custom_kernel_io) if custom_kernel_io else None,
            metadata=custom_kernel_metadata or {}
        )

        # Create comparison request
        request = CompareRequest(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            num_trials=num_trials,
            timeout=timeout,
            atol=atol,
            rtol=rtol
        )

        # Submit job
        job_id = await job_manager.submit_evaluation_job(request)
        logger.info(f"Submitted comparison job: {job_id}")

        # Wait for completion
        result = await job_manager.wait_for_completion(job_id, timeout=timeout)

        if result is None:
            # Job failed or timed out
            job_state = await job_manager.get_job_status(job_id)
            if job_state and job_state.status == "failed":
                error_msg = job_state.error or "Job failed with unknown error"
            else:
                error_msg = f"Job timed out after {timeout} seconds"

            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }

        # Convert result to dict
        result_dict = _convert_result_to_dict(result)

        return result_dict

    @mcp.tool()
    async def validate_kernel(
        kernel_source: str,
        kernel_type: Literal["torch", "torch_cuda", "triton", "cuda", "multi_kernel"],
        io_contract: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Compile and validate a GPU kernel without profiling.

        This tool compiles the kernel and runs validation to check if it executes correctly,
        but skips the expensive profiling phase. Useful for quickly testing if a kernel works
        before doing performance measurements.

        ## VALIDATION PROCESS

        1. **Compilation Check**: Syntax, imports, kernel structure
        2. **Execution Test**: Kernel runs without crashing
        3. **I/O Verification**: Inputs/outputs properly handled

        ## WHY VALIDATION > COMPILE-ONLY

        JIT-compiled kernels (Triton, CUDA) may have runtime errors that only
        appear during execution. Common issues caught by validation:
        - Out-of-bounds memory access
        - Incorrect grid/block dimensions
        - Type mismatches at runtime
        - Missing meta parameters for Triton

        ## KERNEL TYPE REQUIREMENTS

        - **TORCH/TORCH_CUDA**: IOContract optional (auto-generated)
        - **TRITON**: IOContract REQUIRED with meta parameters
        - **CUDA**: IOContract REQUIRED with launch config
        - **MULTI_KERNEL**: IOContract + entry_point in metadata REQUIRED

        ## RESPONSE INTERPRETATION

        - `compiled=true, validated=true`: Kernel works correctly
        - `compiled=true, validated=false`: Compiles but fails at runtime
        - `compiled=false`: Syntax or compilation errors
        - Check error messages for specific issues

        ## EXAMPLE

        ```json
        {
          "kernel_source": "import triton\\nimport triton.language as tl\\n@triton.jit\\ndef kernel(x, BLOCK: tl.constexpr): pass",
          "kernel_type": "triton",
          "io_contract": {
            "args": [
              {"name": "x", "type": "tensor", "role": "input", "tensor_spec": {"shape": [1024], "dtype": "float32", "init": {"kind": "zeros"}}},
              {"name": "BLOCK", "type": "int", "value": 256, "role": "input", "is_meta": true}
            ],
            "launch": {"grid": {"x": 4}, "num_warps": 4}
          }
        }
        ```

        Args:
            kernel_source: The source code of the kernel to validate
            kernel_type: Type of kernel
            io_contract: Input/output contract specification. REQUIRED for TRITON, CUDA, and MULTI_KERNEL types.
                        Optional for TORCH and TORCH_CUDA (auto-generated from get_inputs() if not provided).
                        Defines kernel arguments, tensor specifications, and launch configuration.
            metadata: Kernel metadata for advanced targeting and configuration.
                     Contents depend on kernel_type:
                     - TORCH: {"function_name": "...", "class_name": "...", "method_name": "..."} for targeting specific functions/methods
                     - TRITON: {"kernel_name": "..."} for selecting specific kernel when multiple @triton.jit exist
                     - MULTI_KERNEL: {"entry_point": "..."} REQUIRED - specifies which function to call
                     - CUDA: {"kernel_name": "...", "compile_options": [...]}
            timeout: Maximum time to wait for job completion in seconds (default: 60)

        Returns:
            Compilation and validation status with any error messages
        """
        # Validate IOContract if provided
        is_valid, error_msg = _validate_io_contract(io_contract, kernel_type)

        if not is_valid:
            return {
                "status": "failed",
                "error": f"IOContract validation failed: {error_msg}",
                "compiled": False,
                "validated": False,
                "validation_error": error_msg
            }

        # Build kernel object
        kernel = KernelCode(
            source_code=kernel_source,
            kernel_type=KernelType(kernel_type),
            io=_parse_io_contract(io_contract) if io_contract else None,
            metadata=metadata or {}
        )

        # Create evaluation request with minimal trials (just for validation)
        request = EvaluationRequest(
            kernel=kernel,
            num_trials=1,  # Minimal trials for validation only
            timeout=timeout
        )

        # Submit job
        job_id = await job_manager.submit_evaluation_job(request)
        logger.info(f"Submitted validation job: {job_id}")

        # Wait for completion
        result = await job_manager.wait_for_completion(job_id, timeout=timeout)

        if result is None:
            # Job failed or timed out
            job_state = await job_manager.get_job_status(job_id)
            if job_state and job_state.status == "failed":
                error_msg = job_state.error or "Job failed with unknown error"
            else:
                error_msg = f"Job timed out after {timeout} seconds"

            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg,
                "compiled": False,
                "validated": False
            }

        # Convert result to dict
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = vars(result)

        # Extract key validation information
        kernel_exec_result = result_dict.get('kernel_exec_result', {})

        # Simplify the result for validation-only response
        validation_result = {
            "job_id": job_id,
            "status": "success",
            "compiled": kernel_exec_result.get('compiled', False),
            "validated": kernel_exec_result.get('correctness', False),
            "compilation_error": kernel_exec_result.get('compilation_error'),
            "validation_error": kernel_exec_result.get('validation_error'),
            "compilation_time": kernel_exec_result.get('compilation_time'),
            "metadata": kernel_exec_result.get('metadata', {})
        }

        return validation_result

    @mcp.tool()
    async def get_server_stats() -> Dict[str, Any]:
        """
        Get server health, GPU information, and job statistics.

        Returns current server status, available GPUs, memory usage, and job queue information.

        Returns:
            Server statistics including GPU info and job stats
        """
        import torch

        stats = {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            stats["gpu_info"] = []
            for i in range(torch.cuda.device_count()):
                stats["gpu_info"].append({
                    "device_id": i,
                    "device_name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                    "memory_reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
                    "memory_total_mb": torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                })

        # Get job stats from job manager
        if job_manager:
            job_stats = await job_manager.get_job_stats()
            stats["job_stats"] = job_stats

        return stats

    @mcp.tool()
    async def get_job_status(job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific evaluation job by job_id.

        Returns current status (submitted, compiling, validating, profiling, completed, failed),
        creation time, and any error messages.

        Args:
            job_id: The unique identifier of the job to check

        Returns:
            Job status information
        """
        job_state = await job_manager.get_job_status(job_id)

        if not job_state:
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": f"Job {job_id} not found"
            }

        return {
            "job_id": job_state.job_id,
            "status": job_state.status,
            "created_at": job_state.created_at,
            "error": job_state.error
        }

    logger.info(f"FastMCP server '{mcp.name}' initialized successfully")

    return mcp


# Helper functions (module-level)

def _validate_io_contract(io_dict: Optional[Dict[str, Any]], kernel_type: str) -> tuple[bool, Optional[str]]:
    """
    Validate IOContract structure and provide helpful error messages

    Args:
        io_dict: Dictionary representation of IO contract
        kernel_type: Type of kernel (torch, triton, cuda, etc.)

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
                "See tool description for examples."
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
        if not isinstance(arg, dict):
            return False, f"Argument {i} must be an object/dictionary"

        # Check required fields
        if "name" not in arg:
            return False, f"Argument {i}: missing required 'name' field"

        if "type" not in arg:
            return False, (
                f"Argument '{arg.get('name', i)}': missing required 'type' field. "
                "Must be one of: 'tensor', 'int', 'float', 'str', 'bool'"
            )

        arg_type = arg.get("type")
        if arg_type not in ["tensor", "int", "float", "str", "bool"]:
            return False, (
                f"Argument '{arg.get('name')}': invalid type '{arg_type}'. "
                "Must be one of: 'tensor', 'int', 'float', 'str', 'bool'"
            )

        if "role" not in arg:
            return False, (
                f"Argument '{arg.get('name')}': missing required 'role' field. "
                "Must be one of: 'input', 'output', 'inout'"
            )

        # Validate tensor-specific fields
        if arg_type == "tensor":
            if "tensor_spec" not in arg:
                return False, (
                    f"Tensor argument '{arg.get('name')}': missing required 'tensor_spec'. "
                    "Must include 'shape' (array), 'dtype' (string), and optionally 'init' (for input tensors). "
                    "Example: {'shape': [1024], 'dtype': 'float32', 'init': {'kind': 'randn', 'seed': 42}}"
                )

            tensor_spec = arg.get("tensor_spec", {})
            if "shape" not in tensor_spec:
                return False, f"Tensor '{arg.get('name')}': tensor_spec missing 'shape' (array of integers)"

            if "dtype" not in tensor_spec:
                return False, (
                    f"Tensor '{arg.get('name')}': tensor_spec missing 'dtype'. "
                    "Valid dtypes: float32, float64, float16, bfloat16, int32, int64, int8, uint8, bool"
                )

            # Check init for input/inout tensors
            role = arg.get("role")
            if role in ["input", "inout"] and "init" not in tensor_spec:
                return False, (
                    f"Tensor '{arg.get('name')}' with role='{role}': tensor_spec missing 'init'. "
                    "Input tensors need initialization method. "
                    "Example: {'kind': 'randn', 'seed': 42} or {'kind': 'zeros'}"
                )

            # Validate init if present
            if "init" in tensor_spec:
                init = tensor_spec["init"]
                if "kind" not in init:
                    return False, (
                        f"Tensor '{arg.get('name')}': init missing 'kind'. "
                        "Valid kinds: randn, uniform, zeros, ones, full, arange"
                    )

                kind = init["kind"]
                if kind not in ["randn", "uniform", "zeros", "ones", "full", "arange"]:
                    return False, (
                        f"Tensor '{arg.get('name')}': invalid init kind '{kind}'. "
                        "Valid kinds: randn, uniform, zeros, ones, full, arange"
                    )

                # Validate kind-specific parameters
                if kind == "full" and "fill_value" not in init:
                    return False, f"Tensor '{arg.get('name')}': init kind='full' requires 'fill_value' parameter"

        # Validate scalar values
        elif arg_type in ["int", "float", "str", "bool"]:
            is_meta = arg.get("is_meta", False)
            if not is_meta and "value" not in arg:
                return False, (
                    f"Scalar argument '{arg.get('name')}' (type={arg_type}): missing 'value'. "
                    "Scalars must have 'value' field unless is_meta=true (Triton only)"
                )

    # Validate launch configuration for kernels that need it
    if requires_io_contract and "launch" not in io_dict:
        return False, (
            f"IOContract for {kernel_type.upper()} kernel missing 'launch' configuration. "
            "For Triton: {'grid': {'x': N}, 'num_warps': 4}. "
            "For CUDA: {'grid': {'x': N}, 'block': {'x': 256}}"
        )

    return True, None


def _parse_io_contract(io_dict: Dict[str, Any]) -> IOContract:
    """
    Parse IO contract from dictionary

    Args:
        io_dict: Dictionary representation of IO contract

    Returns:
        IOContract object
    """
    return IOContract.from_dict(io_dict)


def _convert_result_to_dict(result: Any) -> Dict[str, Any]:
    """
    Convert result object to dictionary with proper handling of nested dataclasses

    Args:
        result: Result object from job manager

    Returns:
        Dictionary representation of result
    """
    # Convert result to dict
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump()
    elif hasattr(result, 'dict'):
        result_dict = result.dict()
    else:
        result_dict = vars(result)

    # Convert nested dataclasses to dicts
    if hasattr(result, 'kernel_exec_result') and hasattr(result.kernel_exec_result, 'to_dict'):
        result_dict['kernel_exec_result'] = result.kernel_exec_result.to_dict()

    # Convert ref_runtime for CompareResponse
    if hasattr(result, 'ref_runtime') and result.ref_runtime:
        result_dict['ref_runtime'] = {
            "mean": result.ref_runtime.mean,
            "std": result.ref_runtime.std,
            "min": result.ref_runtime.min,
            "max": result.ref_runtime.max,
            "median": result.ref_runtime.median,
            "percentile_95": result.ref_runtime.percentile_95,
            "percentile_99": result.ref_runtime.percentile_99
        }

    return result_dict
