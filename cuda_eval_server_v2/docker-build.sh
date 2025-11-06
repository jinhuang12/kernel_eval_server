#!/bin/bash

# Build script for CUDA Evaluation Server V2 Docker container
# This script builds the Docker image with all necessary dependencies
# Supports FastAPI, MCP, and hybrid modes

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="cuda-eval-server"
IMAGE_TAG="latest"
DOCKERFILE_PATH="KernelBench/scripts/cuda_eval_server_v2/Dockerfile"
CONTEXT_PATH="."
BUILD_MODE=""  # Optional mode hint for documentation

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --mode)
            BUILD_MODE="$2"
            # Validate mode
            if [[ ! "$BUILD_MODE" =~ ^(fastapi|mcp|both|hybrid)$ ]]; then
                echo -e "${RED}Error: Invalid mode '$BUILD_MODE'${NC}"
                echo "Valid modes: fastapi, mcp, both (or hybrid)"
                exit 1
            fi
            # Normalize hybrid to both
            if [ "$BUILD_MODE" = "hybrid" ]; then
                BUILD_MODE="both"
            fi
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build Docker image for CUDA Evaluation Server V2"
            echo ""
            echo "Options:"
            echo "  --tag TAG        Specify image tag (default: latest)"
            echo "  --mode MODE      Build mode hint (fastapi|mcp|both) - for documentation"
            echo "                   - fastapi: REST API server"
            echo "                   - mcp: MCP server via stdio"
            echo "                   - both: Hybrid mode (FastAPI + MCP)"
            echo "  --no-cache       Build without using cache"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build for any mode (default)"
            echo "  $0 --mode mcp         # Build optimized for MCP"
            echo "  $0 --mode both        # Build for hybrid mode"
            echo "  $0 --tag v2.0         # Build with custom tag"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CUDA Evaluation Server V2 - Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if nvidia-docker/nvidia-container-runtime is available
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected${NC}"
    echo "You may need to install nvidia-docker2 or nvidia-container-toolkit"
    echo "Continuing anyway..."
fi

# Navigate to the parent directory (AIRE-TFL-KernelBench)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../.."

echo -e "${GREEN}Building from directory: $(pwd)${NC}"
echo -e "${GREEN}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
if [ -n "$BUILD_MODE" ]; then
    echo -e "${GREEN}Mode hint: ${BUILD_MODE}${NC}"
fi
echo ""

# Check if requirements file exists
if [ ! -f "requirements.eval_server_v2.txt" ]; then
    echo -e "${RED}Error: requirements.eval_server_v2.txt not found${NC}"
    echo "Expected location: $(pwd)/requirements.eval_server_v2.txt"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE_PATH${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    ${NO_CACHE} \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "$DOCKERFILE_PATH" \
    --progress=plain \
    "$CONTEXT_PATH"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Image created: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "To run the container:"
    echo ""
    if [ -z "$BUILD_MODE" ] || [ "$BUILD_MODE" = "fastapi" ]; then
        echo "  FastAPI mode (REST API):"
        echo "    ./docker-run.sh"
        echo "    ./docker-run.sh --mode fastapi --port 8000"
    fi
    if [ -z "$BUILD_MODE" ] || [ "$BUILD_MODE" = "mcp" ]; then
        echo "  MCP mode (stdio communication):"
        echo "    ./docker-run.sh --mode mcp"
    fi
    if [ -z "$BUILD_MODE" ] || [ "$BUILD_MODE" = "both" ]; then
        echo "  Hybrid mode (FastAPI + MCP):"
        echo "    ./docker-run.sh --mode both --port 8000"
    fi
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose up"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi