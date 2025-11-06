"""
Error handling tests for MCP server
Tests various failure scenarios and edge cases
"""

import pytest
import sys
import os

# Add server directory to path
server_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, server_dir)

from mcp_server import KernelEvalMCPServer


@pytest.mark.asyncio
async def test_missing_io_contract_for_triton():
    """Test that Triton kernels without IO contract are handled with validation error"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": "import triton\n@triton.jit\ndef kernel(): pass",
        "kernel_type": "triton",
        # Missing io_contract - should fail validation immediately
        "num_trials": 10,
        "timeout": 60
    }

    result = await server._evaluate_kernel(args)

    # Should fail validation immediately (new validation logic)
    assert result.get("status") == "failed"
    assert "IOContract validation failed" in result.get("error", "")
    assert result.get("compiled") == False
    assert "REQUIRED" in result.get("validation_error", "")


@pytest.mark.asyncio
async def test_invalid_kernel_type():
    """Test handling of invalid kernel type"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": "print('hello')",
        "kernel_type": "invalid_type",  # Invalid
        "num_trials": 10,
        "timeout": 60
    }

    # Should raise ValueError when creating KernelType enum
    with pytest.raises((ValueError, KeyError)):
        result = await server._evaluate_kernel(args)


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test that timeouts are handled gracefully"""
    server = KernelEvalMCPServer()
    await server.setup()

    # Use a very short timeout
    args = {
        "kernel_source": "import torch\nclass Model(torch.nn.Module):\n  def forward(self, x): return x\ndef get_inputs(): return [torch.randn(1000000, 1000000)]",
        "kernel_type": "torch",
        "num_trials": 1000,  # Many trials
        "timeout": 1  # 1 second timeout
    }

    result = await server._evaluate_kernel(args)

    # Should timeout and return error
    if result.get("status") == "failed":
        assert "timeout" in result.get("error", "").lower() or "timed out" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_empty_kernel_source():
    """Test handling of empty kernel source"""
    server = KernelEvalMCPServer()
    await server.setup()

    args = {
        "kernel_source": "",
        "kernel_type": "torch",
        "num_trials": 10,
        "timeout": 60
    }

    result = await server._evaluate_kernel(args)

    # Should fail compilation
    if "kernel_exec_result" in result:
        exec_result = result["kernel_exec_result"]
        assert exec_result["compiled"] == False


@pytest.mark.asyncio
async def test_malformed_io_contract():
    """Test handling of malformed IO contract with validation"""
    server = KernelEvalMCPServer()
    await server.setup()

    # Malformed IO contract (missing required "type" field)
    args = {
        "kernel_source": "import triton\n@triton.jit\ndef kernel(x): pass",
        "kernel_type": "triton",
        "io_contract": {
            "args": [
                {
                    "name": "x",
                    # Missing "type" field - should fail validation
                    "role": "input"
                }
            ],
            "launch": {"grid": {"x": 4}, "num_warps": 4}
        }
    }

    result = await server._evaluate_kernel(args)

    # Should fail validation with helpful error message
    assert result.get("status") == "failed"
    assert "IOContract validation failed" in result.get("error", "")
    assert "missing required 'type' field" in result.get("validation_error", "").lower()


@pytest.mark.asyncio
async def test_concurrent_tool_calls():
    """Test that server handles concurrent tool calls"""
    server = KernelEvalMCPServer()
    await server.setup()

    # Make multiple concurrent calls
    import asyncio

    tasks = []
    for i in range(3):
        task = server._get_server_stats()
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # All should succeed
    assert len(results) == 3
    for result in results:
        assert result["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
