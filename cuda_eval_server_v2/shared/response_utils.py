"""
Shared response utilities for CUDA Evaluation Server V2
Provides common response conversion and formatting functions
"""

from typing import Dict, Any, Optional
import socket
import os

from shared.models import (
    CompareResponse, EvaluationResponse, KernelExecutionResult,
    RuntimeStats, DeviceMetrics, ComparisonDeviceMetrics
)


def convert_result_to_dict(result: Any) -> Dict[str, Any]:
    """
    Convert a result object to a dictionary, handling both Pydantic models and dataclasses.

    This function is used by both FastAPI and MCP endpoints to ensure consistent
    response formatting.

    Args:
        result: The result object (CompareResponse, EvaluationResponse, etc.)

    Returns:
        Dictionary representation of the result
    """
    # Handle Pydantic models (v2)
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump(exclude_none=True)
    # Handle Pydantic models (v1)
    elif hasattr(result, 'dict'):
        result_dict = result.dict(exclude_none=True)
    # Handle dataclasses or other objects
    elif hasattr(result, '__dict__'):
        result_dict = vars(result)
    else:
        # Already a dict or primitive
        return result

    # Special handling for nested models that might still be dataclasses
    # (for backward compatibility during migration)

    # Handle kernel_exec_result if it has to_dict method (legacy dataclass)
    if 'kernel_exec_result' in result_dict:
        kernel_exec = result_dict['kernel_exec_result']
        if hasattr(kernel_exec, 'to_dict'):
            result_dict['kernel_exec_result'] = kernel_exec.to_dict()
        elif isinstance(kernel_exec, KernelExecutionResult):
            # Already a Pydantic model, ensure it's properly serialized
            result_dict['kernel_exec_result'] = kernel_exec.model_dump(exclude_none=True)

    # Handle ref_runtime for CompareResponse
    if 'ref_runtime' in result_dict:
        ref_runtime = result_dict['ref_runtime']
        if hasattr(ref_runtime, 'model_dump'):
            result_dict['ref_runtime'] = ref_runtime.model_dump(exclude_none=True)
        elif hasattr(ref_runtime, 'to_dict'):
            result_dict['ref_runtime'] = ref_runtime.to_dict()
        elif isinstance(ref_runtime, RuntimeStats):
            # Ensure all fields are present
            result_dict['ref_runtime'] = {
                "mean": ref_runtime.mean,
                "std": ref_runtime.std,
                "min": ref_runtime.min,
                "max": ref_runtime.max,
                "median": ref_runtime.median,
                "percentile_95": ref_runtime.percentile_95,
                "percentile_99": ref_runtime.percentile_99
            }

    return result_dict


def create_error_response(
    error_message: str,
    job_id: Optional[str] = None,
    status_code: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error_message: The error message
        job_id: Optional job ID associated with the error
        status_code: Optional HTTP status code

    Returns:
        Dictionary with error information
    """
    response = {
        "status": "failed",
        "error": error_message,
        "pod_name": socket.gethostname(),
        "pod_ip": socket.gethostbyname(socket.gethostname())
    }

    if job_id:
        response["job_id"] = job_id

    if status_code:
        response["status_code"] = status_code

    return response


def create_no_cache_headers() -> Dict[str, str]:
    """
    Create headers to prevent caching of responses.

    Returns:
        Dictionary of no-cache headers
    """
    return {
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0'
    }


def add_gpu_type_to_response(
    response_dict: Dict[str, Any],
    gpu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add GPU type information to response if available.

    Args:
        response_dict: The response dictionary
        gpu_type: Optional GPU type string

    Returns:
        Updated response dictionary
    """
    if gpu_type and gpu_type != 'unknown':
        response_dict['gpu_type'] = gpu_type

    return response_dict


def format_job_timeout_error(
    job_id: str,
    timeout: int,
    job_status: Optional[str] = None,
    job_error: Optional[str] = None
) -> str:
    """
    Format a timeout error message with job details.

    Args:
        job_id: The job ID
        timeout: The timeout value in seconds
        job_status: Optional job status
        job_error: Optional job error message

    Returns:
        Formatted error message
    """
    if job_status == "failed" and job_error:
        return job_error
    return f"Job {job_id} timed out after {timeout} seconds"