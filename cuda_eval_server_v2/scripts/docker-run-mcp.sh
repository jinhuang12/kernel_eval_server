#!/bin/bash
# Run MCP Docker container

set -e

# Parse arguments
MODE="${1:-mcp}"  # Default to mcp mode
PORT="${2:-8000}"  # Default port for hybrid mode

echo "========================================="
echo "Running MCP Docker Container"
echo "========================================="
echo "Mode: $MODE"

if [ "$MODE" = "mcp" ]; then
    echo "Starting MCP server (stdio mode)"
    echo ""

    docker run --gpus all --rm -it \
      --name cuda-eval-mcp-server \
      -e GPU_DEVICE=0 \
      -e ENABLE_DEVICE_METRICS=false \
      -e LOG_LEVEL=INFO \
      -v /tmp:/tmp \
      cuda-eval-mcp-server:latest \
      python3 main.py --mode mcp --log-level info

elif [ "$MODE" = "hybrid" ]; then
    echo "Starting Hybrid server (FastAPI + MCP)"
    echo "FastAPI available at: http://localhost:$PORT"
    echo ""

    docker run --gpus all --rm -it \
      --name cuda-eval-hybrid-server \
      -p "$PORT:8000" \
      -e GPU_DEVICE=0 \
      -e ENABLE_DEVICE_METRICS=false \
      -e LOG_LEVEL=INFO \
      -v /tmp:/tmp \
      cuda-eval-mcp-server:latest \
      python3 main.py --mode both --host 0.0.0.0 --port 8000 --log-level info

elif [ "$MODE" = "fastapi" ]; then
    echo "Starting FastAPI server only"
    echo "API available at: http://localhost:$PORT"
    echo ""

    docker run --gpus all --rm -it \
      --name cuda-eval-fastapi-server \
      -p "$PORT:8000" \
      -e GPU_DEVICE=0 \
      -e ENABLE_DEVICE_METRICS=false \
      -e LOG_LEVEL=INFO \
      -v /tmp:/tmp \
      cuda-eval-mcp-server:latest \
      python3 main.py --mode fastapi --host 0.0.0.0 --port 8000 --log-level info

else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [mcp|hybrid|fastapi] [port]"
    echo ""
    echo "Examples:"
    echo "  $0 mcp              # Run MCP server only"
    echo "  $0 hybrid 8000      # Run hybrid mode on port 8000"
    echo "  $0 fastapi 8080     # Run FastAPI only on port 8080"
    exit 1
fi
