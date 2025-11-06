#!/bin/bash

# Run script for CUDA Evaluation Server V2 Docker container
# This script starts the Docker container with proper GPU support
# Unified server with both FastAPI REST API and MCP HTTP on single port

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
IMAGE_NAME="cuda-eval-server"
IMAGE_TAG="latest"
CONTAINER_NAME="cuda-eval-server"
HOST_PORT=8000
CONTAINER_PORT=8000
GPU_DEVICE="0"
LOG_LEVEL="info"
DETACHED=false
DEVELOPMENT_MODE=false
NATIVE_MODE=false  # Run without Docker

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            HOST_PORT="$2"
            shift 2
            ;;
        --gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --detach|-d)
            DETACHED=true
            shift
            ;;
        --dev)
            DEVELOPMENT_MODE=true
            shift
            ;;
        --native)
            NATIVE_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run CUDA Evaluation Server V2 (Docker or native)"
            echo "Unified server with FastAPI REST API + MCP HTTP on single port"
            echo ""
            echo "Options:"
            echo "  --port PORT        Host port to bind (default: 8000)"
            echo "  --gpu DEVICE       GPU device to use (default: 0, use 'all' for all GPUs)"
            echo "  --tag TAG          Image tag to use (default: latest)"
            echo "  --name NAME        Container name (default: cuda-eval-server)"
            echo "  --log-level LEVEL  Log level: debug|info|warning|error (default: info)"
            echo "  --detach, -d       Run container in background"
            echo "  --dev              Development mode (mount current directory)"
            echo "  --native           Run without Docker (uses local Python/conda)"
            echo "  --help             Show this help message"
            echo ""
            echo "Endpoints:"
            echo "  REST API:  http://localhost:PORT/health, /evaluate, /compare"
            echo "  MCP HTTP:  http://localhost:PORT/mcp"
            echo ""
            echo "Examples:"
            echo "  $0                        # Start unified server (default)"
            echo "  $0 --port 8080            # Custom port"
            echo "  $0 --gpu all              # Use all GPUs"
            echo "  $0 -d                     # Run in background"
            echo "  $0 --native               # Native mode (no Docker)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CUDA Evaluation Server V2 - Docker Run${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Native mode execution
if [ "$NATIVE_MODE" = true ]; then
    echo -e "${BLUE}Running in native mode (no Docker)${NC}"
    echo ""

    # Navigate to the script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$SCRIPT_DIR"

    # Check if we're on EC2 and activate conda
    if [ -f "/home/ec2-user/miniconda3/bin/conda" ]; then
        echo "Activating conda environment..."
        source /home/ec2-user/miniconda3/bin/conda activate base
    elif [ -f "/home/ubuntu/miniconda3/bin/conda" ]; then
        echo "Activating conda environment..."
        source /home/ubuntu/miniconda3/bin/conda activate base
    fi

    # Set environment variables
    export GPU_DEVICE="$GPU_DEVICE"
    export LOG_LEVEL="$LOG_LEVEL"
    export PYTHONUNBUFFERED=1

    # Build command (unified server with both REST and MCP)
    CMD="python main.py --host 0.0.0.0 --port $HOST_PORT --log-level $LOG_LEVEL"

    # Add reload for dev mode
    if [ "$DEVELOPMENT_MODE" = true ]; then
        CMD="$CMD --reload"
    fi

    echo -e "${GREEN}Executing: $CMD${NC}"
    echo ""

    # Execute
    exec $CMD
    exit $?
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if the image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo -e "${YELLOW}Warning: Image ${IMAGE_NAME}:${IMAGE_TAG} not found${NC}"
    echo "Building image first..."
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    "$SCRIPT_DIR/docker-build.sh" --tag "$IMAGE_TAG"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build image${NC}"
        exit 1
    fi
fi

# Check if nvidia-docker/nvidia-container-runtime is available
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected${NC}"
    echo "The container may not have GPU access"
    echo "Install nvidia-docker2 or nvidia-container-toolkit for GPU support"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stop existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container: ${CONTAINER_NAME}${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Prepare Docker run command
DOCKER_CMD="docker run"

# Determine interactive mode based on detached flag
if [ "$DETACHED" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
else
    DOCKER_CMD="$DOCKER_CMD -it --rm"
fi

# Add basic options
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
DOCKER_CMD="$DOCKER_CMD --user root"
DOCKER_CMD="$DOCKER_CMD --ipc=host"
DOCKER_CMD="$DOCKER_CMD --cap-add=SYS_ADMIN --security-opt seccomp=unconfined"

# Add port mapping (unified server always needs ports)
DOCKER_CMD="$DOCKER_CMD -p ${HOST_PORT}:${CONTAINER_PORT}"

# Add GPU configuration
if [ "$GPU_DEVICE" = "all" ]; then
    DOCKER_CMD="$DOCKER_CMD --gpus all"
else
    DOCKER_CMD="$DOCKER_CMD --gpus $GPU_DEVICE"
fi

# Add environment variables
DOCKER_CMD="$DOCKER_CMD -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"
DOCKER_CMD="$DOCKER_CMD -e PYTHONUNBUFFERED=1"
DOCKER_CMD="$DOCKER_CMD -e ENABLE_DEVICE_METRICS=true"
DOCKER_CMD="$DOCKER_CMD -e LOG_LEVEL=$LOG_LEVEL"

# Add the image
DOCKER_CMD="$DOCKER_CMD ${IMAGE_NAME}:${IMAGE_TAG}"

# Override command (unified server with both REST and MCP)
DOCKER_CMD="$DOCKER_CMD python3 main.py --host 0.0.0.0 --port $CONTAINER_PORT --log-level $LOG_LEVEL"

# Add reload flag in development mode
if [ "$DEVELOPMENT_MODE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD --reload"
fi

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Container: $CONTAINER_NAME"
echo "  Port: $HOST_PORT -> $CONTAINER_PORT"
echo "  GPU: $GPU_DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo "  Detached: $DETACHED"
echo "  Development Mode: $DEVELOPMENT_MODE"
echo ""

# Run the container
echo -e "${GREEN}Starting container...${NC}"
echo "Run CMD: ${DOCKER_CMD}"
eval $DOCKER_CMD

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Container started successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    if [ "$DETACHED" = true ]; then
        echo "Container is running in background"
        echo ""
        echo "Commands:"
        echo "  View logs:    docker logs -f $CONTAINER_NAME"
        echo "  Stop:         docker stop $CONTAINER_NAME"
        echo "  Shell access: docker exec -it $CONTAINER_NAME bash"
        echo ""
    fi

    echo -e "${GREEN}Unified Server Running (FastAPI REST + MCP HTTP)${NC}"
    echo ""
    echo -e "${GREEN}REST API Endpoints:${NC}"
    echo "  Health:   http://localhost:${HOST_PORT}/health"
    echo "  Evaluate: http://localhost:${HOST_PORT}/evaluate"
    echo "  Compare:  http://localhost:${HOST_PORT}/compare"
    echo "  Docs:     http://localhost:${HOST_PORT}/docs"
    echo ""
    echo -e "${GREEN}MCP Endpoint:${NC}"
    echo "  URL:      http://localhost:${HOST_PORT}/mcp"
    echo "  Protocol: Streamable HTTP (JSON-RPC)"
    echo ""

    if [ "$DETACHED" = false ]; then
        echo "Press Ctrl+C to stop the server"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Failed to start container!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi