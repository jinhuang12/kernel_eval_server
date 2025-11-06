"""
Refactored MCP Server for CUDA Evaluation Server V2
Uses Pydantic models directly and shared utilities for consistency with FastAPI
"""

import logging
from typing import Dict, Any


from mcp.server import FastMCP

# Import shared models and utilities
from shared.models import (
    EvaluationRequest, CompareRequest,
    KernelType
)
from shared.request_validator import (
    validate_compare_request, validate_evaluation_request,
    ValidationError
)
from shared.response_utils import (
    convert_result_to_dict, create_error_response,
    format_job_timeout_error
)
from shared.io_contract_validator import raise_if_invalid, IOContractValidationError

# Import job manager and GPU manager
from orchestration.job_manager import JobManager

logger = logging.getLogger(__name__)



def create_mcp_server(job_manager: JobManager) -> FastMCP:
    # Initialize FastMCP server
    mcp = FastMCP(
        name="cuda-eval-server",
        streamable_http_path="/"
    )

    @mcp.tool()
    async def evaluate_kernel(request: EvaluationRequest) -> Dict[str, Any]:
        """
        Compile, validate, and profile a GPU kernel with subprocess isolation for safety.

        Args:
            request: EvaluationRequest with kernel and evaluation parameters

        Returns:
            Dictionary with evaluation results including compilation status,
            correctness, runtime statistics, and any errors.
        """
        if not job_manager:
            return create_error_response("Server not properly initialized")

        try:
            # Validate request parameters
            validate_evaluation_request(request.num_trials, request.timeout)

            # Validate IOContract if present
            if request.kernel.io:
                raise_if_invalid(request.kernel.io, request.kernel.kernel_type)
            elif request.kernel.kernel_type in [KernelType.TRITON, KernelType.CUDA]:
                return create_error_response(
                    f"{request.kernel.kernel_type.value} kernels require io_contract to be specified"
                )

            # Submit job
            job_id = await job_manager.submit_evaluation_job(request)

            # Wait for completion
            result = await job_manager.wait_for_completion(job_id, timeout=request.timeout)

            if result is None:
                # Handle timeout
                job_state = await job_manager.get_job_status(job_id)
                error_msg = format_job_timeout_error(
                    job_id, request.timeout,
                    job_state.status if job_state else None,
                    job_state.error if job_state else None
                )
                return create_error_response(error_msg, job_id)

            # Convert result to dictionary
            result_dict = convert_result_to_dict(result)

            # Add job ID if not present
            if 'job_id' not in result_dict:
                result_dict['job_id'] = job_id

            return result_dict

        except ValidationError as e:
            return create_error_response(f"Validation error: {str(e)}")
        except IOContractValidationError as e:
            return create_error_response(f"IOContract validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in evaluate_kernel: {str(e)}", exc_info=True)
            return create_error_response(f"Evaluation failed: {str(e)}")


    @mcp.tool()
    async def compare_kernels(request: CompareRequest) -> Dict[str, Any]:
        """
        Compare the performance and correctness of two GPU kernels.

        This tool compiles, validates, and profiles both a reference kernel and a custom kernel,
        then compares their outputs for correctness and their execution times for performance.

        Args:
            request: CompareRequest with reference and custom kernels plus parameters

        Returns:
            Dictionary with comparison results including correctness, speedup,
            runtime statistics for both kernels, and any errors.
        """
        if not job_manager:
            return create_error_response("Server not properly initialized")

        try:
            # Validate request parameters
            validate_compare_request(
                request.num_trials, request.timeout,
                request.atol, request.rtol
            )

            # Validate IOContracts
            for kernel, name in [(request.ref_kernel, "reference"), (request.custom_kernel, "custom")]:
                if kernel.io:
                    raise_if_invalid(kernel.io, kernel.kernel_type)
                elif kernel.kernel_type in [KernelType.TRITON, KernelType.CUDA]:
                    return create_error_response(
                        f"{name} kernel ({kernel.kernel_type.value}) requires io_contract to be specified"
                    )
                
            # Submit job for async comparison processing
            job_id = await job_manager.submit_evaluation_job(request)
            
            # Wait for completion with timeout
            result = await job_manager.wait_for_completion(job_id, timeout=request.timeout)

            if result is None:
                # Handle timeout
                job_state = await job_manager.get_job_status(job_id)
                error_msg = format_job_timeout_error(
                    job_id, request.timeout,
                    job_state.status if job_state else None,
                    job_state.error if job_state else None
                )
                return create_error_response(error_msg, job_id)

            # Convert result to dictionary
            result_dict = convert_result_to_dict(result)

            # Add job ID if not present
            if 'job_id' not in result_dict:
                result_dict['job_id'] = job_id

            return result_dict

        except ValidationError as e:
            return create_error_response(f"Validation error: {str(e)}")
        except IOContractValidationError as e:
            return create_error_response(f"IOContract validation error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in compare_kernels: {str(e)}", exc_info=True)
            return create_error_response(f"Comparison failed: {str(e)}")
    
    logger.info(f"FastMCP server '{mcp.name}' initialized successfully")

    return mcp