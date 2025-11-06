"""
FastAPI frontend for CUDA Evaluation Server V2
Provides async HTTP endpoints and orchestrates the compilation/profiling workflow
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import os
import subprocess
from typing import Dict, Optional
from pydantic import ValidationError

from shared.models import CompareRequest, EvaluationRequest, KernelCode, KernelType
from shared.utils import create_error_response, create_no_cache_headers
from shared.metrics_collector import get_metrics_collector
from orchestration.job_manager import JobManager
from mcp_server_refactored import create_mcp_server

logger = logging.getLogger(__name__)

# Global job manager - will be initialized in lifespan
job_manager: JobManager = None
# Global GPU type - detected at startup
gpu_type: str = None
# Global MCP server app - will be initialized in lifespan
mcp_app = None


class ClientError(Exception):
    """Exception for client-side errors that should return HTTP 400"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

def detect_gpu_type() -> str:
    """Detect GPU type from environment or hardware"""
    # First check environment variable
    env_gpu = os.environ.get('GPU_TYPE', '').lower()
    if env_gpu in ['a100', 'h100', 'h200']:
        return env_gpu
    
    # Fallback to hardware detection
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            for gpu in ['a100', 'h100', 'h200']:
                if gpu in gpu_name:
                    return gpu
    except Exception:
        pass
    
    return 'unknown'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI lifespan events
    Initialize services on startup and cleanup on shutdown
    Combines FastAPI lifespan with MCP app lifespan for proper session manager initialization
    """
    global job_manager, gpu_type, mcp_app, logger

    # Startup
    logger.info("Starting CUDA Evaluation Server V2...")

    # Detect GPU type
    gpu_type = detect_gpu_type()
    logger.info(f"Detected GPU type: {gpu_type}")
    
    # Initialize job manager (creates and injects GPU manager into services)
    job_manager = JobManager()
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_jobs_periodically())

    # Initialize MCP server
    logger.info("Initializing MCP server...")
    mcp_server = create_mcp_server(job_manager)
    mcp_app = mcp_server.streamable_http_app()
    logger.info("✅ MCP server initialized at /mcp")

    logger.info("✅ CUDA Evaluation Server V2 started successfully")
    logger.info("   FastAPI REST endpoints: /health, /evaluate, /compare")
    logger.info("   MCP endpoint: /mcp")

    # Enter MCP app's lifespan context to initialize session manager
    async with mcp_app.router.lifespan_context(app):
        yield

    # Shutdown
    logger.info("Shutting down CUDA Evaluation Server V2...")
    cleanup_task.cancel()
    logger.info("✅ CUDA Evaluation Server V2 shutdown complete")


app = FastAPI(
    title="CUDA Evaluation Server V2",
    description="Async CUDA kernel compilation and profiling service using CuPy",
    version="2.0.0",
    lifespan=lifespan
)


# MCP forwarding ASGI app - delegates to mcp_app once initialized
async def mcp_forwarder(scope, receive, send):
    """Forward requests to MCP app once it's initialized"""
    if mcp_app is None:
        # MCP not yet initialized
        await send({
            'type': 'http.response.start',
            'status': 503,
            'headers': [[b'content-type', b'application/json']],
        })
        await send({
            'type': 'http.response.body',
            'body': b'{"error": "MCP server not yet initialized"}',
        })
    else:
        # Forward to MCP app
        await mcp_app(scope, receive, send)


# Mount MCP endpoint
app.mount("/mcp", mcp_forwarder)


async def cleanup_old_jobs_periodically():
    """Background task to periodically clean up old jobs"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            await job_manager.cleanup_old_jobs(max_age_seconds=3600)  # Clean jobs older than 1 hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")


@app.post("/compare", response_model=dict)
async def compare_kernels(request: dict):
    """
    Compare reference and custom kernel performance
    
    This endpoint:
    1. Handles both old (ref_code/custom_code) and new (ref_kernel/custom_kernel) formats
    2. Submits job for async comparison processing
    3. Waits for completion with timeout
    4. Returns comparison results
    """
    try:
        # Check if request is in old format and convert
        if "ref_code" in request and "custom_code" in request:
            # Old format - convert to new format
            logger.info("Converting old request format to new KernelCode format")
            compare_request = CompareRequest(
                ref_kernel=KernelCode(
                    source_code=request["ref_code"],
                    kernel_type=KernelType.TORCH  # Assume reference is PyTorch
                ),
                custom_kernel=KernelCode(
                    source_code=request["custom_code"],
                    kernel_type=KernelType.TORCH_CUDA  # Assume custom is TORCH_CUDA
                ),
                num_trials=request.get("num_trials", 100),
                timeout=request.get("timeout", 30)
            )
        else:
            # New format - parse directly
            compare_request = CompareRequest(**request)
        
        # Business logic validation
        if compare_request.num_trials < 1:
            raise ClientError("num_trials must be at least 1", {"num_trials": compare_request.num_trials})
        
        if compare_request.num_trials > 10000:
            raise ClientError("num_trials cannot exceed 10000", {"num_trials": compare_request.num_trials})
        
        if compare_request.timeout < 5:
            raise ClientError("timeout must be at least 5 seconds", {"timeout": compare_request.timeout})
        
        if compare_request.timeout > 600:
            raise ClientError("timeout cannot exceed 600 seconds", {"timeout": compare_request.timeout})
        
        # Validate Triton kernels have IOContract
        for kernel_name, kernel in [("ref_kernel", compare_request.ref_kernel), 
                                     ("custom_kernel", compare_request.custom_kernel)]:
            if kernel.kernel_type == KernelType.TRITON and not kernel.io:
                raise ClientError(
                    f"Triton kernels require io_contract to be specified",
                    {"kernel": kernel_name, "kernel_type": "triton"}
                )
        
        logger.info(f"Received comparison request: ref_kernel={compare_request.ref_kernel.kernel_type}, "
                   f"custom_kernel={compare_request.custom_kernel.kernel_type}, "
                   f"num_trials={compare_request.num_trials}")
        
        # Submit job for async comparison processing
        job_id = await job_manager.submit_evaluation_job(compare_request)
        
        # Wait for completion with timeout
        result = await job_manager.wait_for_completion(job_id, timeout=compare_request.timeout)
        
        if result is None:
            # Job timed out or failed
            job_state = await job_manager.get_job_status(job_id)
            if job_state and job_state.status == "failed":
                error_msg = job_state.error or "Job failed with unknown error"
            else:
                error_msg = f"Job timed out after {compare_request.timeout} seconds"
            
            error_response = create_error_response(
                error_message=error_msg,
                job_id=job_id
            )
            
            # Return with no-cache headers for consistency with original API
            headers = create_no_cache_headers()
            return JSONResponse(content=error_response, status_code=500, headers=headers)
        
        # Success - convert to dict for response
        # Need to convert nested dataclasses to dicts
        result_dict = result.model_dump()
        
        # Convert kernel_exec_result dataclass to dict
        if hasattr(result.kernel_exec_result, 'to_dict'):
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
        
        # Add GPU type to response
        if gpu_type and gpu_type != 'unknown':
            result_dict['gpu_type'] = gpu_type
        
        # Return with no-cache headers for consistency with original API
        headers = create_no_cache_headers()
        if gpu_type and gpu_type != 'unknown':
            headers['X-GPU-Type'] = gpu_type
        
        return JSONResponse(content=result_dict, status_code=200, headers=headers)
        
    except ValidationError:
        # Re-raise ValidationError to be handled by the exception handler
        raise
    except ClientError:
        # Re-raise ClientError to be handled by the exception handler
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compare_kernels: {e}")
        error_response = create_error_response(
            error_message=f"An unexpected error occurred: {str(e)}"
        )
        headers = create_no_cache_headers()
        return JSONResponse(content=error_response, status_code=500, headers=headers)


@app.post("/", response_model=dict)
async def legacy_evaluate(request: dict):
    """
    Legacy endpoint - redirects to /compare for backward compatibility
    """
    return await compare_kernels(request)


@app.post("/evaluate", response_model=dict)
async def evaluate_kernel(request: dict):
    """
    Evaluate a single kernel's performance
    
    This endpoint:
    1. Compiles and profiles a single kernel
    2. Returns runtime statistics and optional device metrics
    3. No comparison with reference kernel
    """
    try:
        # Parse request as EvaluationRequest
        evaluation_request = EvaluationRequest(**request)
        
        # Business logic validation
        if evaluation_request.num_trials < 1:
            raise ClientError("num_trials must be at least 1", {"num_trials": evaluation_request.num_trials})
        
        if evaluation_request.num_trials > 10000:
            raise ClientError("num_trials cannot exceed 10000", {"num_trials": evaluation_request.num_trials})
        
        if evaluation_request.timeout < 5:
            raise ClientError("timeout must be at least 5 seconds", {"timeout": evaluation_request.timeout})
        
        if evaluation_request.timeout > 600:
            raise ClientError("timeout cannot exceed 600 seconds", {"timeout": evaluation_request.timeout})
        
        # Validate Triton kernels have IOContract
        if evaluation_request.kernel.kernel_type == KernelType.TRITON and not evaluation_request.kernel.io:
            raise ClientError(
                "Triton kernels require io_contract to be specified",
                {"kernel_type": "triton"}
            )
        
        logger.info(f"Received single kernel evaluation request: kernel_type={evaluation_request.kernel.kernel_type}, "
                   f"num_trials={evaluation_request.num_trials}")
        
        # Submit job for async evaluation
        job_id = await job_manager.submit_evaluation_job(evaluation_request)
        
        # Wait for completion with timeout
        result = await job_manager.wait_for_completion(job_id, timeout=evaluation_request.timeout)
        
        if result is None:
            # Job timed out or failed
            job_state = await job_manager.get_job_status(job_id)
            if job_state and job_state.status == "failed":
                error_msg = job_state.error or "Job failed with unknown error"
            else:
                error_msg = f"Job timed out after {evaluation_request.timeout} seconds"
            
            error_response = create_error_response(
                error_message=error_msg,
                job_id=job_id
            )
            
            headers = create_no_cache_headers()
            return JSONResponse(content=error_response, status_code=500, headers=headers)
        
        # Success - convert to dict for response
        # Need to convert nested dataclasses to dicts
        result_dict = result.model_dump()
        
        # Convert kernel_exec_result dataclass to dict
        if hasattr(result.kernel_exec_result, 'to_dict'):
            result_dict['kernel_exec_result'] = result.kernel_exec_result.to_dict()
        
        # Add GPU type to response
        if gpu_type and gpu_type != 'unknown':
            result_dict['gpu_type'] = gpu_type
        
        # Return with no-cache headers
        headers = create_no_cache_headers()
        if gpu_type and gpu_type != 'unknown':
            headers['X-GPU-Type'] = gpu_type
        
        return JSONResponse(content=result_dict, status_code=200, headers=headers)
        
    except ValidationError:
        # Re-raise ValidationError to be handled by the exception handler
        raise
    except ClientError:
        # Re-raise ClientError to be handled by the exception handler
        raise
    except Exception as e:
        logger.error(f"Unexpected error in evaluate_kernel: {e}")
        error_response = create_error_response(
            error_message=f"An unexpected error occurred: {str(e)}"
        )
        headers = create_no_cache_headers()
        return JSONResponse(content=error_response, status_code=500, headers=headers)


@app.get("/health")
async def health_check():
    """Health check endpoint - maintains compatibility with original API"""
    try:
        health_data = await job_manager.health_check()
        return JSONResponse(content=health_data, status_code=200)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy", 
                "error": str(e)
            }, 
            status_code=500
        )


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job - additional endpoint for debugging"""
    job_state = await job_manager.get_job_status(job_id)
    
    if not job_state:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return {
        "job_id": job_state.job_id,
        "status": job_state.status,
        "created_at": job_state.created_at,
        "error": job_state.error
    }


@app.get("/stats")
async def get_server_stats():
    """Get comprehensive server statistics - enhanced with detailed metrics"""
    try:
        # Get comprehensive metrics from the metrics collector
        metrics_collector = get_metrics_collector()
        comprehensive_stats = metrics_collector.get_comprehensive_stats()
        
        # Also include job stats from job manager for completeness
        job_stats = await job_manager.get_job_stats()
        
        # Combine both for a complete view
        return {
            "job_stats": job_stats,
            "compilation_metrics": comprehensive_stats["compilation_metrics"],
            "profiling_metrics": comprehensive_stats["profiling_metrics"], 
            "gpu_utilization": comprehensive_stats["gpu_utilization"],
            "throughput": comprehensive_stats["throughput"],
            "timestamp": comprehensive_stats["timestamp"]
        }
    except Exception as e:
        logger.error(f"Failed to get server stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cleanup-jobs")
async def cleanup_jobs():
    """Manually trigger job cleanup - admin endpoint"""
    try:
        await job_manager.cleanup_old_jobs(max_age_seconds=300)  # Clean jobs older than 5 minutes
        return {"status": "cleanup_completed"}
    except Exception as e:
        logger.error(f"Failed to cleanup jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors as HTTP 400 Bad Request"""
    headers = create_no_cache_headers()
    
    # Format validation errors nicely
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error['loc'])
        errors.append({
            "field": field_path,
            "message": error['msg'],
            "type": error['type']
        })
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid request format",
            "validation_errors": errors,
            "status": "client_error"
        },
        headers=headers
    )


@app.exception_handler(ClientError)
async def client_error_handler(request, exc: ClientError):
    """Handle client errors as HTTP 400"""
    headers = create_no_cache_headers()
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "details": exc.details,
            "status": "client_error"
        },
        headers=headers
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    headers = create_no_cache_headers()
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"},
        headers=headers
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    headers = create_no_cache_headers()
    return JSONResponse(
        status_code=500,
        content=create_error_response(f"Internal server error: {str(exc)}"),
        headers=headers
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
