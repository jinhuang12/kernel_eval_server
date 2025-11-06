"""
Unit tests for MCP server tools
These tests verify individual tool functionality
"""

import pytest
import sys
import os

# Add server directory to path
server_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, server_dir)

from mcp_server import KernelEvalMCPServer
from tests.mcp.fixtures import torch_kernel, triton_kernel


@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """Test that MCP server initializes correctly"""
    server = KernelEvalMCPServer()
    assert server.server is not None
    assert server.job_manager is None  # Not initialized until setup()
    assert server.gpu_manager is None


@pytest.mark.asyncio
async def test_mcp_server_setup():
    """Test that MCP server setup creates job manager"""
    server = KernelEvalMCPServer()
    await server.setup()
    assert server.job_manager is not None
    assert server.gpu_manager is not None


@pytest.mark.asyncio
async def test_get_server_stats():
    """Test get_server_stats tool"""
    server = KernelEvalMCPServer()
    await server.setup()

    stats = await server._get_server_stats()

    assert "status" in stats
    assert stats["status"] == "healthy"
    assert "gpu_available" in stats
    assert "gpu_count" in stats


@pytest.mark.asyncio
async def test_compile_kernel_torch():
    """Test compile_kernel tool with torch kernel"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": torch_kernel.SIMPLE_TORCH_KERNEL,
        "kernel_type": "torch"
    }

    result = await server._compile_kernel(args)

    assert "compiled" in result
    assert "status" in result
    if result["compiled"]:
        assert result["status"] == "success"
        assert result.get("compilation_error") is None
    else:
        assert result["status"] == "compilation_failed"
        assert result.get("compilation_error") is not None


@pytest.mark.asyncio
async def test_get_job_status_not_found():
    """Test get_job_status for non-existent job"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {"job_id": "non-existent-job-id"}
    result = await server._get_job_status(args)

    assert result["status"] == "not_found"
    assert "error" in result


@pytest.mark.asyncio
async def test_parse_io_contract():
    """Test IO contract parsing"""
    server = KernelEvalMCPServer()

    io_dict = triton_kernel.SIMPLE_TRITON_ADD_IO_CONTRACT
    io_contract = server._parse_io_contract(io_dict)

    assert io_contract is not None
    assert len(io_contract.args) == 5
    assert io_contract.launch is not None
    assert io_contract.launch.grid.x == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
