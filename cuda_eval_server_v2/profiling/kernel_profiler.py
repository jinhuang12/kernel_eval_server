"""
Streamlined profiling service - no correctness validation, no pattern logic, no wrapper conditionals
Focus: Profile reference vs C++ wrapper performance only using pre-validated compiled kernel info
"""

import time
import logging
import asyncio
import torch
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from shared.models import BaseExecutableKernel, ProfilingResult
from shared.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)

# Import existing KernelBench functions
try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, kb_dir)
    
    from KernelBench.eval import get_timing_stats, time_execution_with_cuda_event
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    logger.info("Enabled CUDA_LAUNCH_BLOCKING & TORCH_USE_CUDA_DSA for synchronous error detection")
    KB_FUNCTIONS_AVAILABLE = True
    logger.info("KernelBench eval functions imported successfully")
except ImportError as e:
    logger.error(f"Failed to import KernelBench eval functions: {e}")
    KB_FUNCTIONS_AVAILABLE = False


class ProfilingService:
    """
    Profiling service that captures the execution time of reference & custom kernel. 
    Attempts to first capture CUDA graphs to avoid potential overhead from CPU operations,
    falls back to use CUDA events if uncapturable
    """
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
    
    def profile(
        self,
        ref_kernel: BaseExecutableKernel,
        custom_kernel: BaseExecutableKernel,
        num_trials: int = 100,
        job_id: Optional[str] = None,
        gpu_id: Optional[int] = None
    ) -> ProfilingResult:
        """
        Run separated profiling using pre-validated compiled kernel info
        NO correctness validation - assumes compilation service already validated
        
        Args:
            ref_kernel: Kernel to profile
            num_trials: Number of performance measurement trials
            job_id: Optional job identifier
            gpu_id: Optional GPU ID to use (if provided, avoids GPU acquisition)
            
        Returns:
            ProfilingResults with validation and timing data
        """
        profiling_start_time = time.time()
        success = False
        device = torch.device(f"cuda:{gpu_id}")
        
        # Use provided GPU ID directly (no acquisition needed)
        logger.info(f"Job {job_id}: Profiling using provided GPU {gpu_id} (same as compilation)")
        try:
            # Step 1: Profile reference model (try CUDA graphs)
            ref_elapsed_times, ref_method = self._time_with_cuda_graph(
                ref_kernel.with_profiling(use_cuda_graphs=True), *ref_kernel._default_inputs, num_trials=num_trials, device=device
            )
            original_stats = get_timing_stats(ref_elapsed_times, device=device)
            
            logger.info(f"Reference model performance: {original_stats}, profile mode: {ref_method}")
            
            # Run with CUDA graphs if reference kernel executed successfully w/ CUDA graphs 
            custom_kernel = custom_kernel.with_profiling(ref_method == 'cuda_graphs')
            custom_elapsed_times, custom_method = self._time_with_cuda_graph(
                custom_kernel, *ref_kernel._default_inputs, num_trials=num_trials, device=device
            )
            custom_stats = get_timing_stats(custom_elapsed_times, device=device)
            logger.info(f"Custom model performance: {original_stats}")
            
            # Step 3: Aggregate results
            logger.info("Step 3: Aggregating profiling results")
            
            # Calculate speedup
            speedup = original_stats["mean"] / custom_stats["mean"] if custom_stats["mean"] > 0 else 0
            
            metadata = {
                "device": str(device),
                "gpu_id": gpu_id,
                "num_trials": num_trials,
                "speedup": speedup,
                "hardware": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "unknown"
            }
            
            logger.info(f"Profiling completed successfully. Correctness: PASSED (pre-validated), Speedup: {speedup:.2f}x")
            success = True
            
            return ProfilingResult(
                success=True,
                original_runtime=original_stats,
                custom_runtime=custom_stats,
                gpu_id=gpu_id,
                metadata=metadata
            )
                
        except Exception as e:
            logger.error(f"Streamlined profiling failed for job {job_id}: {e}")
            return ProfilingResult(
                success=False,
                error=f"Profiling error: {str(e)}"
            )
        finally:
            # Record profiling completion for metrics
            profiling_time = time.time() - profiling_start_time
            self.metrics_collector.record_profiling_end(
                job_id or "unknown",
                success,
                profiling_time,
                gpu_id if gpu_id is not None else None,
                True if success else None  # correctness
            )
            with device:
                torch.cuda.empty_cache()

    def _time_with_cuda_graph(self, kernel_fn, *args, num_trials=100, device=None):
        """
        Time execution with CUDA graph capture. Falls back to events if graph fails.
        Returns: (elapsed_times, used_graph)
        """        
        elapsed_times = []
        method = 'cuda_graphs'
        
        try:
            if device is None:
                device = torch.cuda.current_device()
            
            # CRITICAL: Set device before any operations
            torch.cuda.set_device(device)
            
            # Move args to device
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            
            # Create static input buffers
            static_args = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    static_buf = torch.empty_like(a, device=device)
                    static_buf.copy_(a)
                    static_args.append(static_buf)
                else:
                    static_args.append(a)
            static_args = tuple(static_args)
            
            # CREATE A NON-DEFAULT STREAM FOR GRAPH CAPTURE
            capture_stream = torch.cuda.Stream(device=device)
            
            # Warmup on the capture stream
            with torch.cuda.stream(capture_stream), torch.no_grad():
                for _ in range(3):
                    _ = kernel_fn(*static_args)
            
            # Wait for warmup to complete
            capture_stream.synchronize()
            
            # Pre-run to allocate output buffers on the capture stream
            with torch.cuda.stream(capture_stream), torch.no_grad():
                static_output = kernel_fn(*static_args)
            
            # Wait for allocation
            capture_stream.synchronize()
            
            # Create graph object
            g = torch.cuda.CUDAGraph()
            
            # Capture the graph on the NON-DEFAULT stream
            with torch.cuda.graph(g, stream=capture_stream), torch.no_grad():
                static_output = kernel_fn(*static_args)
            
            # Synchronize after capture
            torch.cuda.synchronize()
            
            # Create events for timing (can use default stream for replay)
            start_events = []
            end_events = []
            for _ in range(num_trials):
                start_events.append(torch.cuda.Event(enable_timing=True))
                end_events.append(torch.cuda.Event(enable_timing=True))
            
            # Replay the graph on the DEFAULT stream (this is allowed)
            default_stream = torch.cuda.current_stream(device)
            
            for i in range(num_trials):
                # Update inputs if needed
                for s, a in zip(static_args, args):
                    if isinstance(s, torch.Tensor):
                        s.copy_(a)
                
                start_events[i].record(default_stream)
                g.replay()  # Replay on default stream
                end_events[i].record(default_stream)
            
            # Synchronize once at the end
            torch.cuda.synchronize()
            
            # Collect timings
            for i in range(num_trials):
                elapsed_times.append(start_events[i].elapsed_time(end_events[i]))
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.warning(f"Falling back to CUDA events profiling, CUDA Graphs capture failed with: {str(e)}")
            
            # Fallback to regular events
            elapsed_times = time_execution_with_cuda_event(
                kernel_fn, *args, 
                num_trials=num_trials,
                verbose=False,
                device=device
            )
            method = 'cuda_events'
            
        return elapsed_times, method
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for separated profiling service"""
        return {
            "status": "healthy",
            "cuda_available": torch.cuda.is_available(),
            "kb_functions_available": KB_FUNCTIONS_AVAILABLE,
            "total_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "available_gpus": self.gpu_manager.get_available_gpu_count()
        }
