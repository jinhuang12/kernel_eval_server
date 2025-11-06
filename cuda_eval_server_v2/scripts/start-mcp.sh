#!/bin/bash
# Start MCP server only

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SERVER_DIR"

echo "========================================="
echo "Starting MCP Server"
echo "========================================="
echo "Working directory: $SERVER_DIR"
echo ""

# Activate conda environment if on EC2
if [ -f "/home/ubuntu/miniconda3/bin/conda" ]; then
    echo "Activating conda base environment..."
    source /home/ubuntu/miniconda3/bin/conda
    conda activate base
fi

# Start MCP server
python main.py --mode mcp --log-level info

echo ""
echo "MCP server stopped"
