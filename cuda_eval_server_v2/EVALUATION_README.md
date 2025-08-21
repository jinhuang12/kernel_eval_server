# KernelBench Parallel Evaluation Script

This script evaluates all kernels in the `kernelbench_generated_kernels.json` file using `eval_kernel_against_ref` from KernelBench, updating the compile/correct status with actual evaluation results and adding performance metrics.

## Features

- **Parallel GPU Evaluation**: Automatically distributes kernel evaluations across multiple GPUs
- **Checkpoint & Resume**: Can resume from checkpoint if interrupted
- **Performance Metrics**: Captures runtime and calculates speedup
- **Error Resilience**: Continues processing even if individual kernels fail
- **Progress Tracking**: Real-time progress with detailed logging
- **Background Execution**: Can run detached from SSH session

## Quick Start

### Basic Usage

```bash
# Navigate to the script directory
cd AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2/

# Run with all available GPUs (foreground)
python evaluate_all_kernels.py

# Or use the convenience script
./run_evaluation.sh
```

### Run in Background (SSH-safe)

```bash
# Method 1: Using nohup (continues after SSH disconnect)
./run_evaluation.sh --nohup --gpus 0,1,2,3

# Monitor progress
tail -f evaluation_*.log

# Method 2: Using screen (can reattach later)
./run_evaluation.sh --screen --gpus 0,1,2,3

# Reattach to screen session
screen -ls  # List sessions
screen -r kernel_eval_*  # Attach to session
# Press Ctrl+A, D to detach again
```

### Resume from Checkpoint

```bash
# If evaluation was interrupted, resume from checkpoint
python evaluate_all_kernels.py --resume

# Or with the wrapper
./run_evaluation.sh --resume
```

## Command Line Options

### Python Script Options

```bash
python evaluate_all_kernels.py [OPTIONS]

Options:
  --input-json PATH         Input JSON file (default: test_data/kernelbench_generated_kernels.json)
  --output-json PATH        Output JSON file (default: kernelbench_evaluated_{timestamp}.json)
  --gpus GPU_IDS           Comma-separated GPU IDs (default: auto-detect all)
  --checkpoint-interval N   Save checkpoint every N kernels (default: 50)
  --resume                  Resume from checkpoint file
  --num-trials N           Number of performance trials (default: 10)
  --correctness-trials N   Number of correctness trials (default: 1)
  --measure-performance    Enable performance measurement (default: True)
  --verbose                Verbose output
  --start-index N          Start from specific index
  --end-index N            End at specific index
  --build-dir PATH         Directory for build cache (default: build_cache)
```

### Convenience Script Options

```bash
./run_evaluation.sh [OPTIONS]

Options:
  --nohup              Run with nohup in background
  --screen             Run in screen session
  --gpus GPUS          Comma-separated GPU IDs
  --resume             Resume from checkpoint
  --num-trials N       Number of performance trials
  --start-index N      Start from specific index
  --end-index N        End at specific index
  --verbose            Enable verbose output
  --help               Show help
```

## Examples

### Evaluate Specific Range

```bash
# Evaluate kernels 1000-2000 only
python evaluate_all_kernels.py --start-index 1000 --end-index 2000

# Run in background
./run_evaluation.sh --nohup --start-index 1000 --end-index 2000
```

### Use Specific GPUs

```bash
# Use only GPUs 0 and 1
python evaluate_all_kernels.py --gpus 0,1

# With the wrapper script
./run_evaluation.sh --gpus 0,1 --nohup
```

### Verbose Mode with More Trials

```bash
# Verbose output with 20 performance trials
python evaluate_all_kernels.py --verbose --num-trials 20 --correctness-trials 3
```

## Output Format

The script updates the original JSON entries with:

- **Overwrites existing fields**:
  - `compile`: Actual compilation status (true/false)
  - `correct`: Actual correctness status (true/false)

- **Adds new fields**:
  - `ref_runtime_ms`: Reference model runtime in milliseconds
  - `generated_runtime_ms`: Generated kernel runtime in milliseconds
  - `speedup`: Calculated speedup ratio (ref_runtime / generated_runtime)
  - `gpu_id`: Which GPU was used for evaluation
  - `evaluation_timestamp`: When this entry was evaluated
  - `evaluation_error`: Error message if evaluation failed (optional)

### Example Output Entry

```json
{
  "ref": "import torch...",
  "generated": "import torch...",
  "compile": true,
  "correct": true,
  "cuda": true,
  "ref_runtime_ms": 1.234,
  "generated_runtime_ms": 0.567,
  "speedup": 2.176,
  "gpu_id": 0,
  "evaluation_timestamp": "2025-01-08T16:15:00"
}
```

## Monitoring Progress

### Check Logs

```bash
# If running with nohup
tail -f evaluation_*.log

# See summary statistics
grep "EVALUATION SUMMARY" evaluation_*.log -A 10
```

### Check Process Status

```bash
# Find evaluation processes
ps aux | grep evaluate_all_kernels

# Check GPU usage
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi
```

### Checkpoint Files

The script saves progress to `checkpoint_eval.json` periodically. This file contains:
- Processed results so far
- Timestamp of last save
- Can be used to resume if interrupted

## Troubleshooting

### Out of Memory

If you encounter GPU memory issues:
1. Reduce the number of parallel GPU workers
2. Use fewer GPUs: `--gpus 0,1` instead of all
3. Clear GPU cache between evaluations (done automatically)

### Lock Errors

If you see "lock file" errors:
- The script automatically retries once
- These are typically due to concurrent compilation
- Usually resolve themselves

### Resuming After Crash

```bash
# Check if checkpoint exists
ls checkpoint_eval.json

# Resume from checkpoint
python evaluate_all_kernels.py --resume
```

### Killing Background Process

```bash
# Find the process
ps aux | grep evaluate_all_kernels

# Kill by PID
kill <PID>

# Or kill all evaluation processes
pkill -f evaluate_all_kernels
```

## Performance Tips

1. **Use all available GPUs** for maximum throughput
2. **Set checkpoint interval** based on stability (default 50 is good)
3. **Use nohup or screen** for long-running evaluations
4. **Monitor GPU memory** with `nvidia-smi` to ensure no OOM
5. **Build cache** is reused across runs - don't delete unless necessary

## Summary Statistics

After completion, the script prints:
- Total kernels evaluated
- Compilation success rate
- Correctness pass rate
- Average/median/max speedup (if performance measured)

## Files Generated

- `kernelbench_evaluated_YYYYMMDD_HHMMSS.json`: Final results
- `evaluation_YYYYMMDD_HHMMSS.log`: Detailed log (if using wrapper script)
- `checkpoint_eval.json`: Checkpoint file (deleted after successful completion)
- `build_cache/`: Compilation cache directory
