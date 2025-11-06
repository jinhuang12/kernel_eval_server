"""
Job manager for orchestrating evaluation workflow using subprocess workers
Handles GPU acquisition and subprocess management with log streaming
"""

import asyncio
import uuid
import time
import logging
import json
import os
import sys
import subprocess
from typing import Dict, Optional, Any, Union

from shared.models import (
    CompareRequest, CompareResponse, JobState, 
    EvaluationRequest, EvaluationResponse, JobState,
    KernelType, CompilationResult, ValidationResult,
    CompareProfilingResult, ProfilingResult,
    ArgSpec, TensorSpec,
    RuntimeStats, KernelExecutionResult, KernelMetadata, DeviceMetrics, ComparisonDeviceMetrics
)
from shared.utils import HOSTNAME, IP_ADDRESS, create_error_response
from shared.metrics_collector import get_metrics_collector
from shared.gpu_resource_manager import GPUResourceManager
from shared.device_metrics_parser import DeviceMetricsParser, get_device_metrics_parser

logger = logging.getLogger(__name__)


class JobManager:
    """
    Orchestrates the evaluation workflow using subprocess workers
    Manages GPU acquisition and subprocess execution with log streaming
    """
    
    def __init__(self):
        # Only GPU manager stays in main process
        self.gpu_manager = GPUResourceManager()

        self.jobs: Dict[str, JobState] = {}
        self.metrics_collector = get_metrics_collector()
        
        logger.info("JobManager initialized with subprocess-based evaluation")
    
    async def submit_evaluation_job(self, request: Union[EvaluationRequest, CompareRequest]) -> str:
        """
        Submit a new evaluation job and start async processing
        
        Args:
            request: EvaluationRequest
            
        Returns:
            job_id: Unique identifier for tracking the job
        """
        job_id = str(uuid.uuid4())
        
        # Create job state
        job_state = JobState(
            job_id=job_id,
            status="submitted",
            request=request,
            created_at=time.time()
        )
        
        self.jobs[job_id] = job_state
        
        # Record request start for metrics
        self.metrics_collector.record_request_start(job_id)
        
        logger.info(f"Submitted job {job_id}")
        
        # Start async processing without awaiting (fire and forget)
        asyncio.create_task(self._process_evaluation_job(job_id))
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobState]:
        """Get current job status"""
        return self.jobs.get(job_id)
    
    async def wait_for_completion(self, job_id: str, timeout: int = 120) -> Optional[Union[EvaluationResponse, CompareResponse]]:
        """
        Wait for job completion with timeout
        
        Args:
            job_id: Job identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            EvaluationResponse if completed successfully, None if timeout/error
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_state = self.jobs.get(job_id)
            if not job_state:
                logger.error(f"Job {job_id} not found")
                return None
                
            if job_state.status == "completed" and job_state.result:
                return job_state.result
            elif job_state.status == "failed":
                logger.error(f"Job {job_id} failed: {job_state.error}")
                return None
            
            # Poll every 100ms
            await asyncio.sleep(0.1)
        
        # Timeout
        logger.error(f"Job {job_id} timed out after {timeout}s")
        return None
    
    async def _stream_pipe(self, pipe, level, job_id):
        """Stream pipe output to logger"""
        logger_name = f"worker.{job_id}"
        log = logging.getLogger(logger_name)
        
        async for line in pipe:
            line_text = line.decode().strip()
            if line_text:  # Only log non-empty lines
                log.log(level, line_text)
    
    def _get_ncu_command(self, job_id: str) -> Optional[list]:
        """Get NCU command if device metrics are enabled"""
        if os.getenv("ENABLE_DEVICE_METRICS", "false").lower() != "true":
            return None
        essential_vars = ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']
        env_vars = []
        for var in essential_vars:
            value = os.environ.get(var)
            if value:
                env_vars.append(f"{var}={value}")
            
        # Check if NCU is available
        result = subprocess.run(["which", "ncu"], capture_output=True, timeout=5, text=True)
        if result.returncode != 0:
            logger.warning("NCU not found, device metrics will not be collected")
            return None
            
        ncu_path = result.stdout.strip()
        report_path = f"/tmp/ncu_{job_id}.ncu-rep"
        
        # Build NCU command
        ncu_cmd = [ncu_path]
        ncu_cmd.extend([
            "--export", report_path,
            "--nvtx",
            "--set", "full",
            "--nvtx-include", f"{job_id}_original/",
            "--nvtx-include", f"{job_id}_custom/",
            "--target-processes", "application-only", # Needs to be application only in docker container
            "--force-overwrite"
        ])
        
        # Add sections if specified
        ncu_sections = os.getenv("NCU_SECTIONS", "SpeedOfLight,Occupancy,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,LaunchStats,SchedulerStats,WarpStateStats,SourceCounters,SpeedOfLight_RooflineChart")
        if ncu_sections:
            for section in ncu_sections.split(","):
                ncu_cmd.extend(["--section", section.strip()])
        
        return ncu_cmd
    
    def _prepare_subprocess_data(self, request: Any, job_id: str, gpu_id: int, created_at: float, job_type: str = "comparison") -> Dict[str, Any]:
        """Prepare data for subprocess worker"""
        # Use Pydantic v1 compatible serialization
        try:
            # Try Pydantic v2 method first
            subprocess_data = request.model_dump(mode='json')
        except AttributeError:
            # Fall back to Pydantic v1 method
            subprocess_data = request.dict()

        # Add the additional fields needed by subprocess
        subprocess_data.update({
            "job_id": job_id,
            "gpu_id": gpu_id,
            "created_at": created_at,
            "job_type": job_type,
            "server_path": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        })
        
        return subprocess_data
    
    async def _run_subprocess_worker(self, subprocess_data: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """Run subprocess worker with log streaming"""
        log = logging.getLogger(f"worker.{job_id}")
        log.propagate = True  # or False if you attach handlers here
        # Write input data to temp file
        input_file = f"/tmp/job_{job_id}_input.json"
        with open(input_file, 'w') as f:
            json.dump(subprocess_data, f, indent=2)
        
        # Prepare command
        worker_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "subprocess_worker.py"
        )
        cmd = [sys.executable, "-u", worker_script, input_file]
        
        # Wrap with NCU if enabled
        ncu_cmd = self._get_ncu_command(job_id)
        if ncu_cmd:
            cmd = ncu_cmd + cmd
            logger.info(f"Job {job_id}: Running with NCU device metrics collection")
        
        # Set up environment with line buffering
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        logger.info(f"Job {job_id}: Starting subprocess worker")
        
        # Start subprocess with pipe streaming and proper buffering
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            limit=1024*1024  # Increase buffer limit to 1MB
        )
        
        # Stream logs in background
        async for raw in process.stdout:
            # Be gentle with whitespace & encodings
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                log.info(line)  # keep INFO; the tool may write everything to merged stream
        
        # Wait for process completion
        returncode = await process.wait()
        # await asyncio.gather(*log_tasks)
        
        logger.info(f"Job {job_id}: Subprocess completed with exit code {returncode}")
        
        # Always try to read result from temp file, even on crash
        result_file = f"/tmp/job_{job_id}_result.json"
        result = None
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                logger.info(f"Job {job_id}: Successfully read result file with status: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"Job {job_id}: Failed to read result file: {e}")
        
        # Cleanup temp files
        try:
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(result_file):
                os.unlink(result_file)
        except OSError as e:
            logger.warning(f"Job {job_id}: Failed to cleanup temp files: {e}")
        
        # If we got a result (even partial), return it
        if result:
            # If subprocess crashed but we have partial results, mark as completed
            # The partial results will be used to build the response
            if returncode != 0 and result.get("status") not in ["completed", "failed"]:
                logger.warning(f"Job {job_id}: Subprocess crashed (exit code {returncode}) but partial results available")
                result["status"] = "completed"  # Mark as completed so we return partial results
                if not result.get("error"):
                    result["error"] = f"Subprocess terminated unexpectedly (exit code: {returncode})"
            return result
        else:
            # No result file at all - complete failure
            error_msg = f"Subprocess failed with exit code {returncode}"
            if returncode == -9:
                error_msg = "Subprocess was killed (possibly OOM or timeout)"
            elif returncode == -15:
                error_msg = "Subprocess was terminated"
            elif returncode == 1:
                error_msg = "Subprocess failed during initialization"
            
            return {
                "status": "failed",
                "error": error_msg
            }
    
    async def _process_evaluation_job(self, job_id: str):
        """
        Process job using subprocess worker
        
        Args:
            job_id: Job identifier to process
        """
        start_time = time.time()
        success = False
        
        try:
            job_state = self.jobs[job_id]
            request = job_state.request
            job_type = "comparison" if isinstance(request, CompareRequest) else "evaluation"
            
            # Acquire GPU in main process
            async with self.gpu_manager.acquire_gpu(job_id=job_id) as gpu_id:
                logger.info(f"Comparison job {job_id}: Acquired GPU {gpu_id}")
                
                # Prepare subprocess data
                subprocess_data = self._prepare_subprocess_data(
                    request, job_id, gpu_id, job_state.created_at, job_type
                )
                
                # Run subprocess worker
                result = await self._run_subprocess_worker(subprocess_data, job_id)
                
                # Update job state based on result
                job_state.status = result.get("status", "failed")
                
                # Reconstruct CompilationResult from subprocess result
                if result.get("compilation_result"):
                    job_state.compilation_result = CompilationResult(**result["compilation_result"])

                # Reconstruct ValidationResult from subprocess result
                if result.get("validation_result"):
                    job_state.validation_result = ValidationResult(**result["validation_result"])
                      
                # Reconstruct ProfilingResult from subprocess result
                if result.get("profiling_result"):
                    job_state.profiling_result = CompareProfilingResult(**result["profiling_result"]) \
                        if job_type == "comparison" else ProfilingResult(**result["profiling_result"])

                if job_state.status == "failed":
                    job_state.error = result.get("error", "Subprocess execution failed")
                    logger.error(f"Job {job_id}: Failed - {job_state.error}")
                    # Don't return early - we still want to create a response
                    # Set status to completed since we're returning a result
                    job_state.status = "completed"
                
                # Create final EvaluationResponse - handle all cases including compilation/validation failures
                # Build metadata first
                metadata = KernelMetadata(gpu_id=gpu_id)
                if job_state.profiling_result and job_state.profiling_result.metadata:
                    # Add any additional metadata from profiling
                    for key, value in job_state.profiling_result.metadata.items():
                        if key == "kernel_name":
                            metadata.kernel_name = value
                        elif key == "kernel_type":
                            metadata.kernel_type = value
                        elif key == "gpu_type":
                            metadata.gpu_type = value
                        elif key == "cuda_version":
                            metadata.cuda_version = value
                
                # Extract device metrics if available
                device_metrics_parser = DeviceMetricsParser(f"/tmp/ncu_{job_id}.ncu-rep")
                if device_metrics_parser and device_metrics_parser.is_available() and job_id:
                    logger.info("Step 4: Extracting device metrics from NCU report")
                    try:
                        device_metrics_dict = device_metrics_parser.get_metrics_for_request(job_id)
                        if device_metrics_dict:
                            # Check if this is a comparison metrics structure or single metrics
                            if "original_device_metrics" in device_metrics_dict or "custom_device_metrics" in device_metrics_dict:
                                # Comparison metrics - use ComparisonDeviceMetrics
                                metadata.device_metrics = ComparisonDeviceMetrics.from_dict(device_metrics_dict)
                            else:
                                # Single kernel metrics - use DeviceMetrics
                                metadata.device_metrics = DeviceMetrics.from_dict(device_metrics_dict)
                        else:
                            logger.warning(f"No device metrics found for job {job_id}")
                    except Exception as e:
                        logger.warning(f"Failed to extract device metrics: {e}")
                
                # Build RuntimeStats if profiling succeeded
                runtime_stats = None
                runtime = 0.0
                if job_state.profiling_result and job_state.profiling_result.success:
                    if job_type == "comparison":
                        stats_dict = job_state.profiling_result.custom_runtime
                    else:
                        stats_dict = job_state.profiling_result.runtime_stats
                    
                    if stats_dict:
                        runtime = stats_dict.get("mean", 0.0)
                        runtime_stats = RuntimeStats.from_dict(stats_dict)
                
                # Create KernelExecutionResult
                kernel_exec_result = KernelExecutionResult(
                    compiled=job_state.compilation_result.compiles if job_state.compilation_result else False,
                    correctness=job_state.validation_result.is_correct if job_state.validation_result else False,
                    runtime=runtime,
                    metadata=metadata,
                    runtime_stats=runtime_stats,
                    compilation_error=job_state.compilation_result.error if job_state.compilation_result and not job_state.compilation_result.compiles else None,
                    validation_error=job_state.validation_result.error if job_state.validation_result and not job_state.validation_result.is_correct else None
                )
                
                # Create response - always return a response even if compilation/validation failed
                if job_type == "comparison":
                    # Build RuntimeStats for reference kernel
                    ref_runtime = RuntimeStats(mean=0, std=0, min=0, max=0, median=0)
                    if job_state.profiling_result and job_state.profiling_result.original_runtime:
                        ref_runtime = RuntimeStats.from_dict(job_state.profiling_result.original_runtime)
                    
                    job_state.result = CompareResponse(
                        job_id=job_id,
                        kernel_exec_result=kernel_exec_result,
                        ref_runtime=ref_runtime,
                        pod_name=HOSTNAME,
                        pod_ip=IP_ADDRESS,
                        status="success"  # Job completed successfully even if kernel failed
                    )
                else:
                    job_state.result = EvaluationResponse(
                        job_id=job_id,
                        kernel_exec_result=kernel_exec_result,
                        pod_name=HOSTNAME,
                        pod_ip=IP_ADDRESS,
                        status="success"  # Job completed successfully even if kernel failed
                    )
                success = True
                logger.info(f"Job {job_id}: Completed with compiled={kernel_exec_result.compiled}, correctness={kernel_exec_result.correctness}")    
        except Exception as e:
            logger.error(f"Job {job_id}: Unexpected error - {e}")
            job_state = self.jobs.get(job_id)
            if job_state:
                job_state.status = "failed"
                job_state.error = str(e)
        finally:
            # Record request completion for metrics
            end_to_end_time = time.time() - start_time
            self.metrics_collector.record_request_completion(job_id, success, end_to_end_time)
    
    async def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """
        Clean up old completed/failed jobs to prevent memory leaks
        
        Args:
            max_age_seconds: Maximum age of jobs to keep in memory
        """
        current_time = time.time()
        jobs_to_remove = []
        
        for job_id, job_state in self.jobs.items():
            if (current_time - job_state.created_at > max_age_seconds and 
                job_state.status in ["completed", "failed"]):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    async def get_job_stats(self) -> Dict[str, any]:
        """Get statistics about jobs"""
        stats = {
            "total_jobs": len(self.jobs),
            "by_status": {},
            "average_age": 0.0
        }
        
        current_time = time.time()
        total_age = 0.0
        
        for job_state in self.jobs.values():
            status = job_state.status
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            total_age += current_time - job_state.created_at
        
        if self.jobs:
            stats["average_age"] = total_age / len(self.jobs)
        
        return stats
    
    async def health_check(self) -> Dict[str, any]:
        """Health check for job manager"""
        return {
            "status": "healthy",
            "job_stats": await self.get_job_stats(),
            "gpu_manager": {
                "available_gpus": self.gpu_manager.get_available_gpu_count()
            }
        }
