"""
TORCH_CUDA specific validator - handles PyTorch models with embedded CUDA kernels
Uses C++ wrapper transformation and ModelNew execution pattern
"""

import logging
import torch
import sys
from typing import Dict, Any, Optional, List, Sequence, Tuple

from shared.models import ArgSpec, BaseExecutableKernel, ValidationResult
from shared.executable_kernels import TorchCudaExecutableKernel, TorchExecutableKernel
from shared.utils import materialize_tensor
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

def _gather_input_specs(kernel: BaseExecutableKernel) -> List[ArgSpec]:
    specs = getattr(kernel, "input_specs", None)
    if not specs:
        return []
    return [a for a in specs if a.type == "tensor" and getattr(a, "role", "input") in ("input", "inout")]

def _materialize_inputs_from_specs(specs: List[ArgSpec], device: torch.device) -> List[torch.Tensor]:
    inputs: List[torch.Tensor] = []
    for a in specs:
        if a.tensor_spec is None:
            raise RuntimeError(f"Tensor arg '{a.name}' requires tensor_spec for materialization")
        inputs.append(materialize_tensor(a.tensor_spec, device))
    return inputs

def _select_inputs_for_both(
    ref_kernel: BaseExecutableKernel,
    custom_kernel: BaseExecutableKernel,
    device: torch.device
) -> Tuple[List[torch.Tensor], Optional[List[Any]]]:
    """
    Pick one coherent set of inputs to use for both kernels:
    Priority:
      1) ref_kernel._default_inputs (if present)
      2) custom_kernel._default_inputs
      3) materialize from ref_kernel.input_specs
      4) materialize from custom_kernel.input_specs
    Returns (inputs, init_inputs_for_custom_or_none)
    """
    # 1) prefer reference defaults
    if getattr(ref_kernel, "_default_inputs", None):
        ins = list(ref_kernel._default_inputs)
        # ensure device
        ins = [x.to(device) if isinstance(x, torch.Tensor) and x.device != device else x for x in ins]
        return ins, getattr(ref_kernel, "init_inputs", None)

    # 2) custom defaults
    if getattr(custom_kernel, "_default_inputs", None):
        ins = list(custom_kernel._default_inputs)
        ins = [x.to(device) if isinstance(x, torch.Tensor) and x.device != device else x for x in ins]
        return ins, getattr(custom_kernel, "init_inputs", None)

    # 3) ref input_specs
    specs = _gather_input_specs(ref_kernel)
    if specs:
        return _materialize_inputs_from_specs(specs, device), getattr(ref_kernel, "init_inputs", None)

    # 4) custom input_specs
    specs = _gather_input_specs(custom_kernel)
    if specs:
        return _materialize_inputs_from_specs(specs, device), getattr(custom_kernel, "init_inputs", None)

    # Nothing found
    raise RuntimeError("Could not determine inputs for correctness validation; "
                       "provide IOContract or configure _default_inputs on the reference kernel.")


class CorrectnessValidator(BaseKernelValidator):
    """Validator for reference & custom kernel correctness"""
    
    def validate_correctness(
        self,
        ref_kernel: BaseExecutableKernel,
        custom_kernel: BaseExecutableKernel,
        device: torch.device,
        num_correct_trials: int = 1,
        job_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate reference & custom kernel correctness 
        
        Args:
            ref_kernel: Reference kernel
            custom_kernel: Candidate kernel
            device: CUDA device to use
            num_correct_trials: Number of correctness trials
            job_id: Job ID for NVTX range 
            
        Returns:
            ValidationResult with validation results
        """
        torch.cuda.set_device(device)

        # Decide on the inputs we will use for both kernels
        try:
            shared_inputs, ref_init_inputs = _select_inputs_for_both(ref_kernel, custom_kernel, device)
        except Exception as e:
            return ValidationResult(is_correct=False, error=str(e), trials_passed=0, total_trials=num_correct_trials)

        # If the custom kernel needs init_inputs (Torch+CUDA C++ wrapper pattern), pass them through
        if hasattr(custom_kernel, "_set_init_inputs"):
            try:
                custom_kernel._set_init_inputs(ref_init_inputs or [])
            except Exception as e:
                logger.warning(f"_set_init_inputs failed on custom kernel: {e}")

        pass_count = 0
        max_diff_overall = 0.0
        avg_diff_overall = 0.0
        
        # Pre-warm both models once with the same inputs to trigger compilation/JIT
        with torch.no_grad():
            try:
                set_seed(1234)
                _ = ref_kernel(*shared_inputs)
                torch.cuda.synchronize(device)
                _ = custom_kernel(*shared_inputs)
                torch.cuda.synchronize(device)
            except Exception as e:
                return ValidationResult(is_correct=False, error=f"Warmup failed: {e}",
                                        trials_passed=0, total_trials=num_correct_trials)

        
        # Generate seeds for trials
        set_seed(42)
        trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]
        
        for trial in range(num_correct_trials):
            trial_seed = trial_seeds[trial]
            
            # Run reference model
            set_seed(trial_seed)
            if trial > 0 and job_id:
                torch.cuda.nvtx.range_push(f"{job_id}_original")

            ref_output = ref_kernel(*shared_inputs)
            torch.cuda.synchronize(device=device)

            if trial > 0 and job_id:
                torch.cuda.nvtx.range_pop()
                logger.debug(f"NVTX range captured: {job_id}_original/")
            
            # Run CuPy model
            set_seed(trial_seed)
            if trial > 0 and job_id:
                torch.cuda.nvtx.range_push(f"{job_id}_custom")
                
            custom_output = custom_kernel(*shared_inputs)
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
            
            # Check output values with same tolerance as original eval.py
            local_pass = True
            for r, c in zip(R, C):
                if not torch.allclose(r, c, atol=1e-2, rtol=1e-2):
                    local_pass = False
                    diff = (r - c).abs()
                    max_diff_overall = max(max_diff_overall, float(diff.max().item()))
                    avg_diff_overall = max(avg_diff_overall, float(diff.mean().item()))
            if local_pass:
                pass_count += 1

        torch.cuda.empty_cache()

        return ValidationResult(
            is_correct=(pass_count == num_correct_trials),
            trials_passed=pass_count,
            total_trials=num_correct_trials,
            max_difference=(max_diff_overall if max_diff_overall > 0 else None),
            avg_difference=(avg_diff_overall if avg_diff_overall > 0 else None),
        )
