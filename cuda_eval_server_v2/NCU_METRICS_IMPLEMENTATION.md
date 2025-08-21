# NCU Device Metrics Collection Implementation

## Overview

This document describes the implementation of NCU (NVIDIA Nsight Compute) device metrics collection integrated into the CUDA Evaluation Server V2. The system collects detailed GPU performance metrics during kernel validation without requiring the entire server to run under NCU.

## Architecture

The implementation follows a **subprocess-based approach** where NCU profiling is performed during the correctness validation subprocess, keeping the main server process independent of NCU.

```
Main Server Process
    ↓
Compilation Service
    ↓
Subprocess Validator (Two-Pass Execution)
    ├── Pass 1: Regular correctness validation (warmup)
    └── Pass 2: NCU-wrapped validation with NVTX ranges
            ↓
      Device Metrics Extraction
            ↓
      Metrics passed back through CompilationResult
            ↓
      JobManager includes metrics in final response
```

## Key Components

### 1. Enhanced Subprocess Validator (`subprocess_correctness_validator.py`)

- **Two-pass execution**:
  - First pass: Regular correctness validation (serves as warmup)
  - Second pass: Wrapped with NCU for metrics collection (if enabled)
  
- **NCU command detection**:
  - Automatically detects if NCU is available
  - Handles both sudo and non-sudo scenarios
  - Preserves environment variables when using sudo

- **Metrics extraction**:
  - Parses NCU report after subprocess completion
  - Extracts device metrics using the existing DeviceMetricsParser
  - Cleans up temporary NCU report files

### 2. NVTX-Enhanced Subprocess Script (`subprocess_script.py`)

- **NVTX range support**:
  - Wraps original model execution with `request_{job_id}_original` range
  - Wraps custom kernel execution with `request_{job_id}_custom` range
  - Allows NCU to capture metrics for both implementations separately

### 3. Data Flow Integration

- **CompilationResult** extended with `device_metrics` field
- **JobManager** passes device metrics from compilation to final response
- Metrics appear in `kernel_exec_result.metadata.device_metrics`

## Configuration

### Environment Variables

```bash
# Enable device metrics collection
export ENABLE_DEVICE_METRICS=true

# Configure NCU metric sets (optional)
export NCU_METRIC_SETS=speed-of-light  # or "full", "roofline", etc.

# Configure NCU sections (optional)
export NCU_SECTIONS=ComputeWorkloadAnalysis,MemoryWorkloadAnalysis
```

### Running the Server

```bash
# Option 1: Regular server with metrics (if NCU doesn't require sudo)
ENABLE_DEVICE_METRICS=true python main.py

# Option 2: With sudo (preserves environment)
sudo -E ENABLE_DEVICE_METRICS=true python main.py

# Option 3: Using the wrapper script (recommended)
./run_with_ncu.sh
```

## Testing

### Test NCU Metrics Collection

```bash
# Run the comprehensive test
python test_ncu_metrics.py

# Test with running server
ENABLE_DEVICE_METRICS=true python main.py
# In another terminal:
python test_ncu_metrics.py
```

### Expected Response Format

```json
{
  "status": "success",
  "kernel_exec_result": {
    "metadata": {
      "device_metrics": {
        "original_device_metrics": {
          "speed_of_light": {
            "compute_throughput_pct": 85.2,
            "memory_throughput_pct": 92.1
          },
          "detailed_metrics": {
            "l1_hit_rate_pct": 78.9,
            "l2_hit_rate_pct": 89.4
          }
        },
        "custom_device_metrics": {
          "speed_of_light": {
            "compute_throughput_pct": 78.9,
            "memory_throughput_pct": 88.4
          }
        }
      }
    }
  }
}
```

## Advantages of This Approach

1. **Minimal Performance Impact**: 
   - Metrics collection only happens during validation
   - First pass serves as warmup
   - No overhead during normal profiling

2. **Safety**:
   - NCU runs in isolated subprocess
   - Main server unaffected by NCU crashes
   - Graceful degradation if NCU unavailable

3. **Clean Integration**:
   - Reuses existing subprocess infrastructure
   - No changes to profiling pipeline
   - Metrics flow through existing data models

4. **Flexibility**:
   - Can be enabled/disabled via environment variable
   - Works with or without sudo
   - Configurable metric sets and sections

## Edge Cases Handled

1. **NCU Not Available**: Server continues without metrics
2. **Sudo Required**: Automatically detects and uses sudo -E
3. **Subprocess Timeout**: Extended timeout for NCU overhead
4. **Report File Cleanup**: Automatic cleanup of temporary files
5. **Segfault Protection**: Metrics collection in subprocess, isolated from server

## Performance Considerations

- **Overhead**: ~2-3x slower when metrics enabled (due to NCU profiling)
- **Memory**: Minimal - NCU reports are temporary and cleaned up
- **GPU Usage**: Same GPU used for both passes to ensure context consistency

## Troubleshooting

### NCU Not Detected
```bash
# Check NCU installation
ncu --version

# Check permissions
sudo ncu --version
```

### Metrics Not Appearing
1. Check `ENABLE_DEVICE_METRICS=true` is set
2. Verify NCU is accessible
3. Check server logs for NCU-related errors
4. Ensure GPU is available

### Permission Issues
```bash
# Option 1: Run server as root
sudo -E ENABLE_DEVICE_METRICS=true python main.py

# Option 2: Configure NCU for non-root access (system-specific)
```

## Future Enhancements

1. **Caching**: Cache device metrics for identical kernels
2. **Selective Profiling**: Only profile specific kernels based on criteria
3. **Custom Metrics**: Add support for custom NCU metric sets
4. **Metric Aggregation**: Aggregate metrics across multiple runs

## Conclusion

The NCU device metrics collection system provides valuable GPU performance insights without compromising server stability or performance. The subprocess-based approach ensures safety while maintaining clean integration with the existing architecture.
