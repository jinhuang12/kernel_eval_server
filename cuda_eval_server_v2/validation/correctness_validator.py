"""
TORCH_CUDA specific validator - handles PyTorch models with embedded CUDA kernels
Uses C++ wrapper transformation and ModelNew execution pattern
"""

import logging
import torch
import sys
import traceback
from typing import Dict, Any, Optional, List, Sequence, Tuple

from shared.models import BaseExecutableKernel, ValidationResult, IOContract
from shared.executable_kernels import TorchCudaExecutableKernel, TorchExecutableKernel
from io_contract import IOContractManager
from validation.base_validator import BaseKernelValidator

logger = logging.getLogger(__name__)

# Import KernelBench functions
try:
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    sys.path.insert(0, kb_dir)
    
    from KernelBench.eval import (
        
        set_seed,
        
    )
    KB_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import KernelBench eval functions: {e}")
    KB_FUNCTIONS_AVAILABLE = False

def _coerce_to_list(x: Any) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        # only keep tensors
        return [t for t in x if isinstance(t, torch.Tensor)]
    return []

def _same_shapes(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(x.shape == y.shape for x, y in zip(a, b))


def _save_tensor_copies(inputs: List[Any], io_contract: Optional[IOContract]) -> Dict[int, torch.Tensor]:
    """Save copies of output/inout tensors before execution"""
    saved = {}
    if not io_contract or not io_contract.args:
        return saved

    for i, (arg_spec, inp) in enumerate(zip(io_contract.args, inputs)):
        if torch.is_tensor(inp) and arg_spec.role in ("output", "inout"):
            saved[i] = inp.clone()
    return saved


def _restore_tensors(inputs: List[Any], saved_tensors: Dict[int, torch.Tensor]) -> None:
    """Restore tensors to their saved values"""
    for i, saved_tensor in saved_tensors.items():
        inputs[i].copy_(saved_tensor)


def _select_inputs(kernel: BaseExecutableKernel, device: torch.device, default: List = None) -> List:
    """
    Pick one coherent set of inputs to use for both kernels:
    Priority:
      1) kernel._default_inputs (if present)
      2) materialize from kernel.input_specs
    Returns (inputs, init_inputs_for_custom_or_none)
    """
    # 1) use reference defaults 
    if getattr(kernel, "_default_inputs", None):
        ins = list(kernel._default_inputs)
        # ensure device
        ins = [x.to(device) if isinstance(x, torch.Tensor) and x.device != device else x for x in ins]
        return ins
    
    # 2) otherwise io_contract or input_specs
    io_contract = getattr(kernel, "io_contract", None) or getattr(kernel, "input_specs", None)
    if io_contract and hasattr(io_contract, "args"):
        manager = IOContractManager()
        # Generate all inputs, let each kernel type handle its own filtering
        all_inputs = manager.generate_inputs(io_contract, device)
        # For Triton kernels in validation, filter out meta params since they're passed as kwargs
        kernel_type = getattr(kernel, "kernel_type", None)
        if kernel_type and str(kernel_type).lower() == "triton":
            filtered_inputs = [
                inp for i, inp in enumerate(all_inputs)
                if not getattr(io_contract.args[i], "is_meta", False)
            ]
            return filtered_inputs
        return all_inputs
    
    # 3) Use default if exist 
    if default:
        return default
    
    # Nothing found
    raise RuntimeError("Could not determine inputs for execution; "
                       "provide IOContract or configure _default_inputs on the kernel.")


class ExecutableValidator:
    """Validator for input kernel is executable"""

    def validate(
        self,
        kernel: BaseExecutableKernel,
        device: torch.device,
        num_correct_trials: int = 1,
        job_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate kernel can be executed 
        
        Args:
            kernel: Reference kernel
            device: CUDA device to use
            num_correct_trials: Number of correctness trials
            job_id: Job ID for NVTX range 
            
        Returns:
            Is executable
        """    
        torch.cuda.set_device(device)

        # Decide on the inputs we will use for both kernels
        try:
            inputs = _select_inputs(kernel, device)
        except Exception as e:
            return ValidationResult(is_correct=False, error=f"Input selection failed:\n{traceback.format_exc()}", trials_passed=0, total_trials=num_correct_trials)

        pass_count = 0
        
        # Pre-warm both models once with the same inputs to trigger compilation/JIT
        with torch.no_grad():
            try:
                set_seed(1234)
                _ = kernel(*inputs)
                torch.cuda.synchronize(device)
            except Exception as e:
                return ValidationResult(is_correct=False, error=f"Warmup failed:\n{traceback.format_exc()}",
                                        trials_passed=0, total_trials=num_correct_trials)

        
        # Generate seeds for trials
        set_seed(42)
        trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]
        
        last_error = None  # Track last error for potential reporting
        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]
            
            # Run kernel
            try:
                set_seed(trial_seed)
                if trial > 0 and job_id:
                    torch.cuda.nvtx.range_push(f"{job_id}_original")

                ref_output = kernel(*inputs)
                torch.cuda.synchronize(device=device)

                if trial > 0 and job_id:
                    torch.cuda.nvtx.range_pop()
                    logger.debug(f"NVTX range captured: {job_id}_original/")
                
                # Execution succeeded (regardless of return value)
                pass_count += 1
            except Exception as e:
                error_msg = f"Trial {trial} failed:\n{traceback.format_exc()}"
                logger.warning(error_msg)
                # Store the last error for potential return
                last_error = error_msg
                # Continue to next trial
                
        torch.cuda.empty_cache()

        # Ensure we have an error message if validation failed
        error_msg = last_error
        if pass_count < num_correct_trials and not error_msg:
            # Fallback error message if none was captured
            error_msg = f"Execution failed: {pass_count}/{num_correct_trials} trials passed"

        return ValidationResult(
            is_correct=(pass_count == num_correct_trials),
            trials_passed=pass_count,
            total_trials=num_correct_trials,
            error=error_msg
        )

class CorrectnessValidator(BaseKernelValidator):
    """Validator for reference & custom kernel correctness"""
    
    def validate_correctness(
        self,
        ref_kernel: BaseExecutableKernel,
        custom_kernel: BaseExecutableKernel,
        device: torch.device,
        num_correct_trials: int = 1,
        job_id: Optional[str] = None,
        atol: float = 1e-2,
        rtol: float = 1e-2
    ) -> ValidationResult:
        """
        Validate reference & custom kernel correctness

        Args:
            ref_kernel: Reference kernel
            custom_kernel: Candidate kernel
            device: CUDA device to use
            num_correct_trials: Number of correctness trials
            job_id: Job ID for NVTX range
            atol: Absolute tolerance for numerical comparison (default: 1e-2)
            rtol: Relative tolerance for numerical comparison (default: 1e-2)

        Returns:
            ValidationResult with validation results
        """
        torch.cuda.set_device(device)

        # Log when custom tolerances are used (different from defaults)
        DEFAULT_ATOL = 1e-2
        DEFAULT_RTOL = 1e-2
        if atol != DEFAULT_ATOL or rtol != DEFAULT_RTOL:
            logger.info(f"Using custom tolerances: atol={atol}, rtol={rtol} (defaults: atol={DEFAULT_ATOL}, rtol={DEFAULT_RTOL})")

        # Decide on the inputs we will use for both kernels
        try:
            ref_inputs = _select_inputs(ref_kernel, device)
            custom_inputs = _select_inputs(custom_kernel, device, ref_inputs)
            ref_init_inputs = getattr(ref_kernel, "init_inputs", None) 
            custom_init_inputs = getattr(custom_kernel, "init_inputs", None) or ref_init_inputs

        except Exception as e:
            return ValidationResult(is_correct=False, error=f"Input selection failed:\n{traceback.format_exc()}", trials_passed=0, total_trials=num_correct_trials)

        # If the custom kernel needs init_inputs (Torch+CUDA C++ wrapper pattern), pass them through
        if hasattr(custom_kernel, "_set_init_inputs"):
            try:
                custom_kernel._set_init_inputs(custom_init_inputs or [])
            except Exception as e:
                logger.warning(f"_set_init_inputs failed on custom kernel:\n{traceback.format_exc()}")
        # Same for _default_inputs for (Torch+CUDA C++ wrapper pattern)
        if not getattr(custom_kernel, "_default_inputs", None):
            custom_kernel._default_inputs = custom_inputs

        # Save original tensor values to restore before each execution
        # This prevents accumulation when kernels use atomic operations (e.g., atomic_add with SPLIT_K > 1)
        ref_saved_tensors = _save_tensor_copies(ref_inputs, ref_kernel.io_contract)
        custom_saved_tensors = _save_tensor_copies(custom_inputs, custom_kernel.io_contract)

        pass_count = 0
        max_diff_overall = 0.0
        avg_diff_overall = 0.0
        
        # Pre-warm both models once with the same inputs to trigger compilation/JIT
        with torch.no_grad():
            try:
                set_seed(1234)
                # Restore tensors to original values before warmup
                _restore_tensors(ref_inputs, ref_saved_tensors)
                _restore_tensors(custom_inputs, custom_saved_tensors)
                _ = ref_kernel(*ref_inputs)
                torch.cuda.synchronize(device)
                _ = custom_kernel(*custom_inputs)
                torch.cuda.synchronize(device)
            except Exception as e:
                return ValidationResult(is_correct=False, error=f"Warmup failed:\n{traceback.format_exc()}",
                                        trials_passed=0, total_trials=num_correct_trials)

        
        # Generate seeds for trials
        set_seed(42)
        trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]
        
        last_error = None  # Track last error for potential reporting
        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]

            # Restore tensors to original values before each trial
            _restore_tensors(ref_inputs, ref_saved_tensors)
            _restore_tensors(custom_inputs, custom_saved_tensors)

            # Run reference model
            set_seed(trial_seed)
            if trial > 0 and job_id:
                torch.cuda.nvtx.range_push(f"{job_id}_original")

            ref_output = ref_kernel(*ref_inputs)
            torch.cuda.synchronize(device=device)

            if trial > 0 and job_id:
                torch.cuda.nvtx.range_pop()
                logger.debug(f"NVTX range captured: {job_id}_original/")
            
            # Run CuPy model
            set_seed(trial_seed)
            if trial > 0 and job_id:
                torch.cuda.nvtx.range_push(f"{job_id}_custom")
                
            custom_output = custom_kernel(*custom_inputs)
            torch.cuda.synchronize(device=device)

            if trial > 0 and job_id:
                torch.cuda.nvtx.range_pop()
                logger.debug(f"NVTX range captured: {job_id}_custom/")
            
            R = _coerce_to_list(ref_output)
            C = _coerce_to_list(custom_output)

            # Check output shapes
            if not _same_shapes(R, C):
                error_msg = f"Shape mismatch: ref={[r.shape for r in R]}, custom={[c.shape for c in C]}"
                return ValidationResult(
                    is_correct=False,
                    error=error_msg,
                    trials_passed=pass_count,
                    total_trials=num_correct_trials
                )
            
            # Check output values with specified tolerance
            local_pass = True
            for r, c in zip(R, C):
                if not torch.allclose(r, c, atol=atol, rtol=rtol):
                    local_pass = False
                    diff = (r - c).abs()
                    max_diff_overall = max(max_diff_overall, float(diff.max().item()))
                    avg_diff_overall = max(avg_diff_overall, float(diff.mean().item()))
            if local_pass:
                pass_count += 1

        torch.cuda.empty_cache()

        # Construct error message if validation failed
        error_msg = None
        if pass_count < num_correct_trials:
            # Validation failed - provide detailed error message
            if max_diff_overall > 0:
                error_msg = (
                    f"Numerical outputs don't match within tolerance. "
                    f"Trials passed: {pass_count}/{num_correct_trials}. "
                    f"Max difference: {max_diff_overall:.6e}, "
                    f"Avg difference: {avg_diff_overall:.6e}. "
                    f"Tolerance: atol={atol}, rtol={rtol}"
                )
            else:
                # Validation failed but no difference tracked (shouldn't happen, but handle gracefully)
                error_msg = f"Validation failed: {pass_count}/{num_correct_trials} trials passed"

        return ValidationResult(
            is_correct=(pass_count == num_correct_trials),
            trials_passed=pass_count,
            total_trials=num_correct_trials,
            max_difference=(max_diff_overall if max_diff_overall > 0 else None),
            avg_difference=(avg_diff_overall if avg_diff_overall > 0 else None),
            error=error_msg,
        )
