"""
Streamlined profiling service - no correctness validation, no pattern logic, no wrapper conditionals
Focus: Profile reference vs C++ wrapper performance only using pre-validated compiled kernel info
"""

import time
import logging
import asyncio
import torch
import traceback
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from shared.models import BaseExecutableKernel, ProfilingResult, CompareProfilingResult
from shared.executable_kernels import MultiKernelExecutableKernel
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
    
    def _calculate_runtime_stats(self, elapsed_times: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive runtime statistics including percentiles.
        
        Args:
            elapsed_times: List of elapsed times in milliseconds
            
        Returns:
            Dict containing mean, std, min, max, median, percentile_95, percentile_99
            All timings are in milliseconds
        """
        times_array = np.array(elapsed_times)
        
        stats = {
            "mean": float(np.mean(times_array)),
            "std": float(np.std(times_array)),
            "min": float(np.min(times_array)),
            "max": float(np.max(times_array)),
            "median": float(np.median(times_array)),
            "percentile_95": float(np.percentile(times_array, 95)),
            "percentile_99": float(np.percentile(times_array, 99))
        }
        
        # Round to 3 significant figures for consistency
        for key in stats:
            stats[key] = round(stats[key], 3)
        
        return stats
    
    def profile(
        self,
        kernel: BaseExecutableKernel,
        job_id: str,
        num_trials: int = 100,
        gpu_id: Optional[int] = None,
        profiling_method: str = "cuda_graphs",
    ) -> ProfilingResult:
        """
        Profile a single kernel's performance
        
        Args:
            kernel: Kernel to profile
            job_id: Job identifier
            num_trials: Number of performance measurement trials
            gpu_id: Optional GPU ID to use
            
        Returns:
            ProfilingResult with runtime statistics
        """
        profiling_start_time = time.time()
        success = False
        device = torch.device(f"cuda:{gpu_id}")
        
        # Use provided GPU ID directly
        logger.info(f"Job {job_id}: Profiling kernel using GPU {gpu_id}")

        # Auto-detect multi_kernel types and use cuda_events to avoid allocator corruption
        # Multi-kernel types often contain non-capturable operations (torch.unique, torch.sort, etc.)
        if isinstance(kernel, MultiKernelExecutableKernel) and profiling_method == "cuda_graphs":
            logger.info(
                f"Job {job_id}: Multi-kernel detected, automatically using cuda_events "
                f"to avoid non-capturable operations that would corrupt CUDA allocator"
            )
            profiling_method = "cuda_events"

        profile_with_cuda_graphs = profiling_method == "cuda_graphs"
        try:
            # Profile the kernel (try CUDA graphs first)
            if profile_with_cuda_graphs:
                elapsed_times, method = self._time_with_cuda_graph(
                    kernel.with_profiling(use_cuda_graphs=profile_with_cuda_graphs), *kernel._default_inputs, 
                    num_trials=num_trials, device=device
                )
            else:
                elapsed_times = time_execution_with_cuda_event(
                    kernel.with_profiling(use_cuda_graphs=profile_with_cuda_graphs), *kernel._default_inputs, 
                    num_trials=num_trials, verbose=False, device=device
                )
                method = 'cuda_events'

            # Use our comprehensive stats calculation
            runtime_stats = self._calculate_runtime_stats(elapsed_times)
            
            # Get hardware and device info separately for metadata
            hardware = torch.cuda.get_device_name(device=device) if device else "unknown"
            device_str = str(device)
            
            logger.info(f"Kernel performance: {runtime_stats}, profile mode: {method}")
            
            metadata = {
                "device": device_str,
                "gpu_id": gpu_id,
                "num_trials": num_trials,
                "profiling_method": method,
                "hardware": hardware
            }
            
            logger.info(f"Profiling completed successfully")
            success = True
            
            return ProfilingResult(
                success=True,
                runtime_stats=runtime_stats,
                gpu_id=gpu_id,
                metadata=metadata
            )
                
        except Exception as e:
            logger.error(f"Profiling failed for job {job_id}: {e}")
            return ProfilingResult(
                success=False,
                runtime_stats={},
                error=f"Profiling error:\n{traceback.format_exc()}"
            )
        finally:
            # Record profiling completion for metrics
            profiling_time = time.time() - profiling_start_time
            self.metrics_collector.record_profiling_end(
                job_id or "unknown",
                success,
                profiling_time,
                gpu_id if gpu_id is not None else None,
                None  # No correctness check for single kernel
            )
            with device:
                torch.cuda.empty_cache()
    
    def compare_profile(
        self,
        ref_kernel: BaseExecutableKernel,
        custom_kernel: BaseExecutableKernel,
        job_id: str,
        num_trials: int = 100,
        gpu_id: Optional[int] = None
    ) -> CompareProfilingResult:
        """
        Compare two kernels by profiling each and calculating speedup
        
        Args:
            ref_kernel: Reference kernel to profile
            custom_kernel: Custom kernel to profile
            job_id: Job identifier
            num_trials: Number of performance measurement trials
            gpu_id: Optional GPU ID to use
            
        Returns:
            CompareProfilingResult with comparison metrics
        """
        profiling_start_time = time.time()
        success = False

        # Auto-detect if either kernel is multi_kernel and use cuda_events for both
        # This ensures fair comparison and avoids allocator corruption
        forced_profiling_method = None
        if isinstance(ref_kernel, MultiKernelExecutableKernel) or isinstance(custom_kernel, MultiKernelExecutableKernel):
            logger.info(
                f"Job {job_id}: Multi-kernel detected in comparison, "
                f"using cuda_events for both kernels for fair comparison"
            )
            forced_profiling_method = "cuda_events"

        try:
            # Profile reference kernel
            ref_result = self.profile(
                ref_kernel, job_id, num_trials, gpu_id,
                profiling_method=forced_profiling_method or "cuda_graphs"
            )

            if not ref_result.success:
                return CompareProfilingResult(
                    success=False,
                    error=f"Reference kernel profiling failed: {ref_result.error}"
                )

            # Use same profiling method for custom kernel (either forced or from ref result)
            custom_profiling_method = forced_profiling_method or ref_result.metadata.get('profiling_method', 'cuda_graphs')
            # Profile custom kernel
            custom_result = self.profile(
                custom_kernel, job_id, num_trials, gpu_id,
                profiling_method=custom_profiling_method
            )
            
            if not custom_result.success:
                return CompareProfilingResult(
                    success=False,
                    error=f"Custom kernel profiling failed: {custom_result.error}"
                )
            
            # Calculate speedup
            ref_mean = ref_result.runtime_stats.get("mean", 0)
            custom_mean = custom_result.runtime_stats.get("mean", 0)
            speedup = ref_mean / custom_mean if custom_mean > 0 else 0
            
            # Combine metadata
            combined_metadata = {
                **ref_result.metadata,
                "speedup": speedup,
                "ref_profiling_method": ref_result.metadata.get("profiling_method"),
                "custom_profiling_method": custom_result.metadata.get("profiling_method")
            }
            
            logger.info(f"Comparison profiling completed. Speedup: {speedup:.2f}x")
            success = True
            
            return CompareProfilingResult(
                success=True,
                original_runtime=ref_result.runtime_stats,
                custom_runtime=custom_result.runtime_stats,
                speedup=speedup,
                gpu_id=gpu_id,
                metadata=combined_metadata
            )
            
        except Exception as e:
            logger.error(f"Comparison profiling failed for job {job_id}: {e}")
            return CompareProfilingResult(
                success=False,
                error=f"Comparison profiling error:\n{traceback.format_exc()}"
            )
        finally:
            # Record profiling completion for metrics
            profiling_time = time.time() - profiling_start_time
            self.metrics_collector.record_profiling_end(
                job_id or "unknown",
                success,
                profiling_time,
                gpu_id if gpu_id is not None else None,
                True if success else None  # correctness for comparison
            )

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
            traceback.print_exc()
            logger.error(f"CUDA Graphs capture failed: {str(e)}")

            # CRITICAL: When CUDA graph capture fails, PyTorch's allocator enters a corrupted
            # state where captures_underway counter is not properly reset. This is a known
            # PyTorch limitation with no public API to fix it.
            #
            # The allocator will reject ALL subsequent GPU memory allocations in this process,
            # making fallback to CUDA events impossible.
            #
            # Note: Multi-kernel types are automatically profiled with cuda_events to avoid
            # this issue. If this error occurs, it's likely a non-multi-kernel type that
            # contains non-capturable operations.

            logger.error(
                f"Cannot fallback to CUDA events after failed graph capture - "
                f"CUDA allocator is corrupted. Kernel contains non-capturable operations "
                f"like torch.unique(), torch.sort(), torch.nonzero(), etc."
            )

            raise RuntimeError(
                f"CUDA graph capture failed: {str(e)}. "
                f"Kernel contains non-capturable operations. "
                f"This shouldn't happen for multi_kernel types (auto-detected). "
                f"For other kernel types, use profiling_method='cuda_events'."
            ) from e
            
        return elapsed_times, method
    
    def _capture_device_metrics(self, exec_kernel, inputs, job_id: str, kind: str):
        """
        Profile exactly ONE replay under graphs using NVTX delimiters.
        Parent NCU is launched with --nvtx-include f"{job_id}_{kind}/graph_profile" 
        and --graph-profiling node, so this single replay is captured.
        """
        tag = f"{job_id}_{kind}/"
        torch.cuda.nvtx.range_push(tag)
        exec_kernel(*inputs)   # single replay is profiled
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for profiling service"""
        return {
            "status": "healthy",
            "cuda_available": torch.cuda.is_available(),
            "kb_functions_available": KB_FUNCTIONS_AVAILABLE,
            "total_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
