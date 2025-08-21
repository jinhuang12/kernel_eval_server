#!/bin/bash

# Script to run kernel evaluation in background
# Usage:
#   ./run_evaluation.sh                    # Run with defaults
#   ./run_evaluation.sh --gpus 0,1,2,3    # Specify GPUs
#   ./run_evaluation.sh --resume           # Resume from checkpoint
#   ./run_evaluation.sh --nohup           # Run with nohup in background
#   ./run_evaluation.sh --screen          # Run in screen session

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
PYTHON_CMD="python3"
SCRIPT_PATH="evaluate_all_kernels.py"
LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"
RUN_MODE="foreground"
EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup)
            RUN_MODE="nohup"
            shift
            ;;
        --screen)
            RUN_MODE="screen"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --nohup              Run with nohup in background"
            echo "  --screen             Run in screen session"
            echo "  --gpus GPUS          Comma-separated GPU IDs (e.g., 0,1,2,3)"
            echo "  --resume             Resume from checkpoint"
            echo "  --num-trials N       Number of performance trials (default: 10)"
            echo "  --start-index N      Start from specific index"
            echo "  --end-index N        End at specific index"
            echo "  --verbose            Enable verbose output"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run in foreground with all GPUs"
            echo "  $0 --nohup --gpus 0,1,2,3   # Run in background with specific GPUs"
            echo "  $0 --screen --resume         # Resume in screen session"
            echo "  $0 --nohup --start-index 1000 --end-index 2000  # Evaluate specific range"
            exit 0
            ;;
        *)
            # Pass through other arguments
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: $SCRIPT_PATH not found!${NC}"
    exit 1
fi

# Check for CUDA availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA might not be available.${NC}"
    exit 1
fi

# Display GPU information
echo -e "${GREEN}Available GPUs:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s (%s MB)\n", $1, $2, $3}'
echo ""

# Run based on mode
case $RUN_MODE in
    nohup)
        echo -e "${GREEN}Starting evaluation with nohup...${NC}"
        echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
        echo -e "${YELLOW}Command: nohup $PYTHON_CMD $SCRIPT_PATH $EXTRA_ARGS > $LOG_FILE 2>&1 &${NC}"
        
        nohup $PYTHON_CMD $SCRIPT_PATH $EXTRA_ARGS > $LOG_FILE 2>&1 &
        PID=$!
        
        echo -e "${GREEN}Process started with PID: $PID${NC}"
        echo ""
        echo "Monitor progress with:"
        echo "  tail -f $LOG_FILE"
        echo ""
        echo "Check if still running:"
        echo "  ps -p $PID"
        echo ""
        echo "Stop the process:"
        echo "  kill $PID"
        ;;
        
    screen)
        SESSION_NAME="kernel_eval_$(date +%Y%m%d_%H%M%S)"
        echo -e "${GREEN}Starting evaluation in screen session: $SESSION_NAME${NC}"
        echo -e "${YELLOW}Command: $PYTHON_CMD $SCRIPT_PATH $EXTRA_ARGS${NC}"
        
        screen -dmS $SESSION_NAME $PYTHON_CMD $SCRIPT_PATH $EXTRA_ARGS
        
        echo -e "${GREEN}Screen session started: $SESSION_NAME${NC}"
        echo ""
        echo "Attach to session:"
        echo "  screen -r $SESSION_NAME"
        echo ""
        echo "Detach from session:"
        echo "  Press Ctrl+A, then D"
        echo ""
        echo "List sessions:"
        echo "  screen -ls"
        echo ""
        echo "Kill session:"
        echo "  screen -X -S $SESSION_NAME quit"
        ;;
        
    foreground)
        echo -e "${GREEN}Starting evaluation in foreground...${NC}"
        echo -e "${YELLOW}Command: $PYTHON_CMD $SCRIPT_PATH $EXTRA_ARGS${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        $PYTHON_CMD $SCRIPT_PATH $EXTRA_ARGS
        ;;
esac
