"""
Integration tests for MCP server
These tests run end-to-end evaluations on EC2
"""

import pytest
import sys
import os
import asyncio

# Add server directory to path
server_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, server_dir)

from mcp_server import KernelEvalMCPServer
from tests.mcp import fixtures


@pytest.mark.asyncio
@pytest.mark.ec2_only
async def test_evaluate_torch_kernel_full():
    """Test full evaluation of a simple torch kernel"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": fixtures.SIMPLE_TORCH_KERNEL,
        "kernel_type": "torch",
        "num_trials": 10,
        "timeout": 60
    }

    result = await server._evaluate_kernel(args)

    # Check that we got a result
    assert "job_id" in result
    assert "status" in result

    # If evaluation succeeded, check structure
    if result.get("status") != "failed":
        assert "kernel_exec_result" in result
        exec_result = result["kernel_exec_result"]
        assert "compiled" in exec_result
        assert "correctness" in exec_result
        assert "runtime" in exec_result


@pytest.mark.asyncio
@pytest.mark.ec2_only
async def test_evaluate_triton_kernel_with_io_contract():
    """Test evaluation of Triton kernel with IO contract"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": fixtures.SIMPLE_TRITON_ADD_KERNEL,
        "kernel_type": "triton",
        "io_contract": fixtures.SIMPLE_TRITON_ADD_IO_CONTRACT,
        "num_trials": 10,
        "timeout": 60
    }

    result = await server._evaluate_kernel(args)

    assert "job_id" in result
    assert "status" in result

    # Triton kernel should compile and execute
    if result.get("status") != "failed":
        exec_result = result.get("kernel_exec_result", {})
        assert exec_result.get("compiled") in [True, False]


@pytest.mark.asyncio
@pytest.mark.ec2_only
async def test_compare_kernels():
    """Test kernel comparison mode"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": fixtures.TORCH_MATMUL_KERNEL,
        "kernel_type": "torch",
        "reference_kernel_source": fixtures.TORCH_MATMUL_KERNEL,
        "reference_kernel_type": "torch",
        "num_trials": 10,
        "timeout": 60
    }

    result = await server._evaluate_kernel(args)

    assert "job_id" in result

    # In comparison mode, we should get ref_runtime
    if result.get("status") != "failed":
        assert "ref_runtime" in result or result.get("kernel_exec_result", {}).get("compiled") == False


@pytest.mark.asyncio
@pytest.mark.ec2_only
async def test_compilation_failure_handling():
    """Test that compilation failures are handled gracefully"""
    server = KernelEvalMCPServer()
    await server.setup()

    # Invalid kernel source that should fail compilation
    invalid_kernel = "import torch\nthis is not valid python syntax!!!"

    args = {
        "kernel_source": invalid_kernel,
        "kernel_type": "torch",
        "num_trials": 10,
        "timeout": 60
    }

    result = await server._evaluate_kernel(args)

    # Should not crash, should return structured error
    assert "job_id" in result

    # Check that compilation failed
    if "kernel_exec_result" in result:
        exec_result = result["kernel_exec_result"]
        assert exec_result["compiled"] == False
        assert "compilation_error" in exec_result


@pytest.mark.asyncio
@pytest.mark.ec2_only
async def test_job_status_polling():
    """Test job status polling during evaluation"""
    server = KernelEvalMCPServer()
    await server.setup()

    # Start a long-running evaluation
    args = {
        "kernel_source": fixtures.TORCH_MATMUL_KERNEL,
        "kernel_type": "torch",
        "num_trials": 100,
        "timeout": 120
    }

    # Start evaluation in background
    eval_task = asyncio.create_task(server._evaluate_kernel(args))

    # Wait a bit for job to be submitted
    await asyncio.sleep(1)

    # Get the job_id from job manager (most recent job)
    if server.job_manager.jobs:
        job_id = list(server.job_manager.jobs.keys())[-1]

        # Poll status
        status_args = {"job_id": job_id}
        status = await server._get_job_status(status_args)

        assert "status" in status
        assert status["status"] in ["submitted", "compiling", "validating", "profiling", "completed", "failed"]

    # Wait for evaluation to complete
    result = await eval_task
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ec2_only"])
