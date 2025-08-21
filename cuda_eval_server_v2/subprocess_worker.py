#!/usr/bin/env python3
"""
Subprocess worker that handles the complete evaluation pipeline:
- Compilation
- Validation 
- Profiling
Runs in complete isolation from the parent process
"""

import sys
import json
import os
import logging
import traceback
import torch
from dataclasses import asdict
from typing import Dict, Any, Optional

# Setup sys.path BEFORE any local imports
if len(sys.argv) == 2:
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    server_path = input_data.get("server_path")
    if server_path and server_path not in sys.path:
        sys.path.insert(0, server_path)

# Now we can import using absolute imports
from compilation.compiler_service import CompilationService
from profiling.kernel_profiler import ProfilingService
from validation import CorrectnessValidator

# Set up logging to stdout for parent process to capture
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def save_state(job_state: Dict[str, Any], job_id: str):
    """Save current job state to temp file"""
    result_file = f"/tmp/job_{job_id}_result.json"
    with open(result_file, 'w') as f:
        # Convert non-serializable objects to dicts
        serializable_state = {
            "job_id": job_state["job_id"],
            "status": job_state["status"],
            "created_at": job_state["created_at"],
            "error": job_state["error"]
        }
        
        # Add compilation result if present
        if job_state["compilation_result"]:
            comp_result = job_state["compilation_result"]
            serializable_state["compilation_result"] = {
                "compiles": comp_result.compiles,
                "compilation_time": comp_result.compilation_time,
                "error": comp_result.error
            }

        # Add validation result if present
        if job_state["validation_result"]:
            val_result = job_state["validation_result"]
            serializable_state["validation_result"] = val_result
        
        # Add profiling result if present
        if job_state["profiling_result"]:
            prof_result = job_state["profiling_result"]
            serializable_state["profiling_result"] = prof_result
        
        json.dump(serializable_state, f, indent=2)
    

def run_evaluation_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline in subprocess
    
    Args:
        input_data: Job data from parent process
        
    Returns:
        JobState-compatible dictionary with results
    """
    job_id = input_data.get('job_id', 'unknown')
    gpu_id = input_data['gpu_id']
    
    # Initialize JobState-compatible result structure
    job_state = {
        "job_id": job_id,
        "status": "started",
        "created_at": input_data.get('created_at'),
        "compilation_result": None,
        "validation_result": None,
        "profiling_result": None,
        "error": None
    }
    
    # Save initial state
    save_state(job_state, job_id)
    
    try:
        # Set GPU device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            logger.info(f"Set CUDA device to: {device}")
        else:
            raise RuntimeError("CUDA not available in subprocess")
        
        # Reconstruct request objects from input data
        from shared.models import (
            KernelCode, KernelType, CompilationRequest, 
            IOContract, ProfilingResult, ArgSpec, TensorSpec
        )
        
        # Reconstruct KernelCode objects
        ref_kernel = KernelCode.from_dict(input_data['ref_kernel'])
        custom_kernel = KernelCode.from_dict(input_data['custom_kernel'])

        # Create compilation requests
        # Reference kernel
        ref_compilation_request = CompilationRequest(
            kernel_code=ref_kernel,
            job_id=job_id
        )
        # Custom kernel
        custom_compilation_request = CompilationRequest(
            kernel_code=custom_kernel,
            job_id=job_id
        )
        
        # ===== STEP 1: COMPILATION =====
        logger.info(f"Starting compilation for job {job_id}")
        job_state["status"] = "compiling"
        save_state(job_state, job_id)
        
        # Initialize compilation service (no GPU manager needed - GPU already set)
        compilation_service = CompilationService()
        
        # Compile kernels
        ref_compilation_result = compilation_service.compile_kernel(ref_compilation_request, gpu_id)
        custom_compilation_result = compilation_service.compile_kernel(custom_compilation_request, gpu_id)
        job_state["compilation_result"] = custom_compilation_result
        
        if not custom_compilation_result.compiles:
            job_state["status"] = "failed"
            job_state["error"] = f"Compilation failed: {custom_compilation_result.error}"
            save_state(job_state, job_id)
            return job_state
        
        logger.info(f"Compilation successful!")
    
        # ===== STEP 2: VALIDATION =====
        logger.info(f"Starting correctness check for job {job_id}")
        job_state["status"] = "validating"
        save_state(job_state, job_id)
        validator = CorrectnessValidator()

        validation_result = validator.validate_correctness(
            ref_kernel=ref_compilation_result.kernel,
            custom_kernel=custom_compilation_result.kernel,
            device=device,
            num_correct_trials=2,
            job_id=job_id
        )
        job_state["validation_result"] = asdict(validation_result)

        if not validation_result.is_correct:
            job_state["status"] = "failed"
            job_state["error"] = f"Validation failed: {validation_result.error}"
            save_state(job_state, job_id)
            return job_state
        
        logger.info(f"Correctness check validation successful!")

        # ===== STEP 3: PROFILING =====
        logger.info(f"Starting profiling for job {job_id}")
        job_state["status"] = "profiling"
        save_state(job_state, job_id)
        
        
        # Create profiling service with dummy GPU manager
        profiling_service = ProfilingService()
        
        # Profile the kernels
        num_trials = input_data.get('num_trials', 100)
        
        # Create ProfilingResult object
        profiling_result = profiling_service.profile(
            ref_kernel=ref_compilation_result.kernel,
            custom_kernel=custom_compilation_result.kernel,
            num_trials=num_trials,
            job_id=job_id,
            gpu_id=gpu_id
        )
        
        job_state["profiling_result"] = asdict(profiling_result)
        
        if not profiling_result.success:
            job_state["status"] = "failed"
            job_state["error"] = f"Profiling failed: {profiling_result.error}"
            save_state(job_state, job_id)
            return job_state
        
        # ===== SUCCESS =====
        logger.info(f"Evaluation completed successfully for job {job_id}")
        job_state["status"] = "completed"
        save_state(job_state, job_id)
        
        return job_state
        
    except Exception as e:
        logger.error(f"Subprocess error: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        job_state["status"] = "failed"
        job_state["error"] = f"Subprocess execution error: {str(e)}"
        save_state(job_state, job_id)
        
        return job_state


def main():
    """Main entry point for subprocess worker"""
    try:
        # Load input data from file
        if len(sys.argv) != 2:
            raise ValueError("Usage: subprocess_worker.py <input_file>")
        
        input_file = sys.argv[1]
        logger.info(f"Loading input from: {input_file}")
        
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Path already set up at module import time
        job_state = run_evaluation_pipeline(input_data)
        
        # Final save of state
        save_state(job_state, job_state["job_id"])
        
        # Exit with appropriate code
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in subprocess worker: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
