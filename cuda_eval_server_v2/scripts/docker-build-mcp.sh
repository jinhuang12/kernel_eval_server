#!/bin/bash
# Build MCP Docker image

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SERVER_DIR")")")"

cd "$REPO_ROOT"

echo "========================================="
echo "Building MCP Docker Image"
echo "========================================="
echo "Repository root: $REPO_ROOT"
echo "Dockerfile: KernelBench/scripts/cuda_eval_server_v2/Dockerfile.mcp"
echo ""

# Build the image
docker build \
  -f KernelBench/scripts/cuda_eval_server_v2/Dockerfile.mcp \
  -t cuda-eval-mcp-server:latest \
  .

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo "Image: cuda-eval-mcp-server:latest"
echo ""
echo "To run:"
echo "  MCP mode:     docker run --gpus all --rm -it cuda-eval-mcp-server:latest"
echo "  Hybrid mode:  docker run --gpus all --rm -it -p 8000:8000 cuda-eval-mcp-server:latest python3 main.py --mode both --host 0.0.0.0 --port 8000"
echo ""
