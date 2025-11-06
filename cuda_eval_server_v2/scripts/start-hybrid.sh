#!/bin/bash
# Start both FastAPI and MCP servers (hybrid mode)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SERVER_DIR"

echo "========================================="
echo "Starting Hybrid Server (FastAPI + MCP)"
echo "========================================="
echo "Working directory: $SERVER_DIR"
echo "FastAPI: http://localhost:8000"
echo "MCP: stdio protocol"
echo ""

# Activate conda environment if on EC2
if [ -f "/home/ubuntu/miniconda3/bin/conda" ]; then
    echo "Activating conda base environment..."
    source /home/ubuntu/miniconda3/bin/conda
    conda activate base
fi

# Start hybrid mode (FastAPI in background, MCP in foreground)
python main.py --mode both --host 0.0.0.0 --port 8000 --log-level info

echo ""
echo "Hybrid server stopped"
