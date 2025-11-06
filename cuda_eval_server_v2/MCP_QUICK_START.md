# MCP Server - Quick Start Guide

## 🚀 Quick Setup on EC2

```bash
# SSH to EC2
ssh p5e-cmh

# Activate conda environment
/home/ec2-user/miniconda3/bin/conda activate base

# Navigate to server directory
cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2

# Install MCP dependency
pip install mcp>=0.9.0

# Test the MCP server
python test_mcp_server.py
```

## 🔧 Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cuda-eval-server": {
      "command": "ssh",
      "args": [
        "p5e-cmh",
        "cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2 && /home/ubuntu/miniconda3/bin/conda run -n base python main.py --mode mcp --log-level info"
      ]
    }
  }
}
```

Restart Claude Desktop to activate.

## ✅ Test It Works

Ask Claude to evaluate a kernel:

```
Can you evaluate this PyTorch kernel for me?

import torch

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs():
    return [torch.randn(32, 32)]
```

💡 **Quick Tip:** For fast iteration during development, use the `validate_kernel` tool instead of `evaluate_kernel`. It compiles and validates your kernel without the overhead of profiling, making it much faster for testing if your kernel works correctly.

## 📚 Full Documentation

See `MCP_README.md` for complete documentation.