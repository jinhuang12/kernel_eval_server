"""
FastAPI frontend for CUDA Evaluation Server V2
Provides async HTTP endpoints and orchestrates the compilation/profiling workflow
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio

from shared.models import EvaluationRequest, KernelCode, KernelType
from shared.utils import create_error_response, create_no_cache_headers
from shared.metrics_collector import get_metrics_collector
from orchestration.job_manager import JobManager

logger = logging.getLogger(__name__)

# Global job manager - will be initialized in lifespan
job_manager: JobManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI lifespan events
    Initialize services on startup and cleanup on shutdown
    """
    global job_manager, logger
    
    # Startup
    logger.info("Starting CUDA Evaluation Server V2...")
    
    # Initialize job manager (creates and injects GPU manager into services)
    job_manager = JobManager()
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_jobs_periodically())
    
    logger.info("✅ CUDA Evaluation Server V2 started successfully")
    
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


@app.post("/", response_model=dict)
async def evaluate_kernel(request: dict):
    """
    Evaluate CUDA kernel - main endpoint with backward compatibility
    
    This endpoint:
    1. Handles both old (ref_code/custom_code) and new (ref_kernel/custom_kernel) formats
    2. Submits job for async processing
    3. Waits for completion with timeout
    4. Returns results in original API format
    """
    try:
        # Check if request is in old format and convert
        if "ref_code" in request and "custom_code" in request:
            # Old format - convert to new format
            logger.info("Converting old request format to new KernelCode format")
            evaluation_request = EvaluationRequest(
                ref_kernel=KernelCode(
                    source_code=request["ref_code"],
                    kernel_type=KernelType.TORCH  # Assume reference is PyTorch
                ),
                custom_kernel=KernelCode(
                    source_code=request["custom_code"],
                    kernel_type=KernelType.TORCH_CUDA  # Assume custom is TORCH_CUDA
                ),
                num_trials=request.get("num_trials", 100),
                timeout=request.get("timeout", 120)
            )
        else:
            # New format - parse directly
            evaluation_request = EvaluationRequest(**request)
        
        logger.info(f"Received evaluation request: ref_kernel={evaluation_request.ref_kernel.kernel_type}, "
                   f"custom_kernel={evaluation_request.custom_kernel.kernel_type}, "
                   f"num_trials={evaluation_request.num_trials}")
        
        # Submit job for async processing
        job_id = await job_manager.submit_evaluation_job(evaluation_request)
        
        # Wait for completion with timeout
        result = await job_manager.wait_for_completion(job_id, timeout=evaluation_request.timeout)
        
        if result is None:
            # Job timed out or failed
            job_state = await job_manager.get_job_status(job_id)
            if job_state and job_state.status == "failed":
                error_msg = job_state.error or "Job failed with unknown error"
            else:
                error_msg = f"Job timed out after {request.timeout} seconds"
            
            error_response = create_error_response(
                error_message=error_msg,
                job_id=job_id
            )
            
            # Return with no-cache headers for consistency with original API
            headers = create_no_cache_headers()
            return JSONResponse(content=error_response, status_code=500, headers=headers)
        
        # Success - convert to dict for response
        result_dict = result.dict()
        
        # Return with no-cache headers for consistency with original API
        headers = create_no_cache_headers()
        return JSONResponse(content=result_dict, status_code=200, headers=headers)
        
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
