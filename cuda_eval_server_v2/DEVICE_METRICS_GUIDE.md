# Device Metrics Collection Guide

## Overview

The CUDA Evaluation Server V2 now supports capturing detailed device metrics during kernel profiling using NVIDIA NCU (Nsight Compute). This feature provides deep insights into GPU performance bottlenecks by collecting Speed of Light metrics, cache performance, pipeline utilization, and occupancy analysis for both reference PyTorch models and custom CUDA kernels.

## Recent Updates (August 2025)

- **Fixed NCU metric extraction**: Corrected all metric names to match actual NCU output
- **Enhanced metric categories**: Now extracting 6 comprehensive metric categories
- **Improved error handling**: Better handling of empty or missing metrics
- **Accurate metric mapping**: Proper mapping between NCU metrics and simplified names

## Features

### 📊 Comprehensive Device Metrics

The system now extracts six categories of metrics:

1. **Speed of Light Analysis**: Overall GPU utilization efficiency
2. **Detailed Performance Metrics**: Cache hit rates, IPC, warp occupancy
3. **Memory Subsystem Metrics**: DRAM bandwidth, L2 throughput
4. **Compute Utilization Metrics**: Pipeline utilization, occupancy limiters
5. **Pipeline Activity Metrics**: FMA, ALU, Tensor core active cycles
6. **Occupancy Analysis**: Block/grid configuration, waves per SM

### 🔍 Key NCU Metrics Collected

#### Primary Indicators (Speed of Light)
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` - Overall GPU throughput
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` - SM compute throughput
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` - Memory bandwidth utilization
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` - GPU DRAM throughput

#### Cache Performance
- `l1tex__t_sector_hit_rate.pct` - L1 cache hit rate
- `lts__t_sector_hit_rate.pct` - L2 cache hit rate
- `lts__throughput.avg.pct_of_peak_sustained_elapsed` - L2 throughput

#### Compute Metrics
- `sm__warps_active.avg.pct_of_peak_sustained_active` - Warp occupancy
- `sm__inst_executed.avg.per_cycle_active` - Instructions per cycle (IPC)
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` - SM active cycles

#### Pipeline Utilization
- `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` - FMA pipeline
- `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed` - ALU pipeline
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` - Tensor cores
- `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed` - Shared memory
- `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed` - FP64 pipeline

#### Occupancy Limiters
- `launch__occupancy_limit_registers` - Register pressure limit
- `launch__occupancy_limit_shared_mem` - Shared memory limit
- `launch__occupancy_limit_warps` - Warp/block size limit
- `launch__occupancy_limit_blocks` - Block limit
- `launch__waves_per_multiprocessor` - Waves per SM (tail effects)

## Quick Start

### 1. Prerequisites

Ensure you have NVIDIA Nsight Compute installed:

```bash
# Check if NCU is available
ncu --version

# If not installed, download from:
# https://developer.nvidia.com/nsight-compute
```

### 2. Enable Device Metrics Collection

Set the environment variable to enable metrics:

```bash
export ENABLE_DEVICE_METRICS=true
```

### 3. Launch Server with NCU

Use the provided wrapper script to launch the server with NCU profiling:

```bash
# Navigate to server directory
cd AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2/

# Basic launch with default settings
./run_with_ncu.sh

# Custom configuration with specific sections
./run_with_ncu.sh --port 8080 --ncu-sections "SpeedOfLight,Occupancy,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis"
```

### 4. Send Evaluation Requests

The API remains unchanged - device metrics are automatically captured:

```bash
curl -X POST "http://localhost:8000/" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_code": "...",
    "custom_code": "...",
    "num_trials": 100
  }'
```

### 5. Access Device Metrics

Device metrics are included in the response payload with the corrected structure:

```json
{
  "status": "success",
  "kernel_exec_result": {
    "metadata": {
      "device_metrics": {
        "original_device_metrics": {
          "speed_of_light": {
            "compute_throughput_pct": 85.2,
            "memory_throughput_pct": 92.1,
            "compute_memory_throughput_pct": 88.5,
            "sm_throughput_pct": 82.3,
            "dram_throughput_pct": 91.8,
            "gpu_dram_throughput_pct": 90.2
          },
          "detailed_metrics": {
            "l1_hit_rate_pct": 89.2,
            "l2_hit_rate_pct": 73.4,
            "warp_occupancy_pct": 78.9,
            "sm_active_cycles_pct": 92.1,
            "instructions_per_cycle": 2.34,
            "waves_per_sm": 3.2
          },
          "memory_metrics": {
            "dram_avg_bandwidth_gb_s": 456.2,
            "dram_total_bandwidth_gb_s": 512.8,
            "dram_active_cycles_pct": 88.4,
            "l1_writeback_active_pct": 45.2,
            "l1_read_sectors_pct": 67.8,
            "l2_throughput_pct": 82.3
          },
          "compute_metrics": {
            "fma_pipe_utilization_pct": 75.2,
            "fp64_pipe_utilization_pct": 12.3,
            "alu_pipe_utilization_pct": 68.9,
            "xu_pipe_utilization_pct": 34.5,
            "tensor_pipe_utilization_pct": 0.0,
            "instructions_per_cycle": 2.34,
            "occupancy_limit_blocks": 16,
            "occupancy_limit_registers": 65536,
            "occupancy_limit_shared_mem": 49152,
            "occupancy_limit_warps": 64,
            "registers_per_thread": 32
          },
          "pipeline_metrics": {
            "fma_pipe_active_pct": 72.3,
            "alu_pipe_active_pct": 65.4,
            "tensor_pipe_active_pct": 0.0,
            "shared_pipe_active_pct": 45.6,
            "fp64_pipe_active_pct": 10.2,
            "sm_issue_active_pct": 88.9
          },
          "occupancy_metrics": {
            "occupancy_limit_registers": 65536,
            "occupancy_limit_shared_mem": 49152,
            "occupancy_limit_warps": 64,
            "occupancy_limit_blocks": 16,
            "waves_per_sm": 3.2,
            "block_size": 256,
            "grid_size": 1024,
            "shared_mem_per_block": 4096
          }
        },
        "custom_device_metrics": {
          "speed_of_light": {
            "compute_throughput_pct": 78.9,
            "memory_throughput_pct": 88.4,
            "compute_memory_throughput_pct": 82.1,
            "sm_throughput_pct": 76.5,
            "dram_throughput_pct": 87.2,
            "gpu_dram_throughput_pct": 86.8
          }
          // ... similar structure for other metric categories
        }
      }
    }
  }
}
```

## Configuration Options

### Environment Variables

```bash
# Enable device metrics collection
export ENABLE_DEVICE_METRICS=true

# Specify NCU sections to collect
export NCU_SECTIONS="SpeedOfLight,Occupancy,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis"

# Set NCU report output directory
export NCU_REPORT_DIR="/tmp"
```

### NCU Sections for Optimal Metrics

| Section | Metrics Provided | Performance Impact |
|---------|------------------|-------------------|
| `SpeedOfLight` | Overall GPU utilization | Low (~2x overhead) |
| `Occupancy` | Thread/warp/block utilization | Low |
| `ComputeWorkloadAnalysis` | Pipeline utilization, IPC | Medium (~3x overhead) |
| `MemoryWorkloadAnalysis` | Cache hits, bandwidth | Medium |
| `InstructionStats` | Instruction mix analysis | High (~5x overhead) |
| `SchedulerStats` | Warp scheduling efficiency | High |

### Recommended Section Combinations

```bash
# Fast profiling (minimal overhead)
export NCU_SECTIONS="SpeedOfLight,Occupancy"

# Balanced profiling (good insights, moderate overhead)
export NCU_SECTIONS="SpeedOfLight,Occupancy,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis"

# Comprehensive profiling (maximum insights, high overhead)
export NCU_SECTIONS="SpeedOfLight,Occupancy,ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,InstructionStats,SchedulerStats"
```

## Understanding Device Metrics

### Speed of Light Analysis

The Speed of Light (SOL) analysis shows how efficiently your kernel uses available GPU resources:

| Metric | Description | Optimization Target |
|--------|-------------|-------------------|
| `compute_throughput_pct` | Overall compute utilization | > 80% for compute-bound |
| `memory_throughput_pct` | Memory bandwidth utilization | > 80% for memory-bound |
| `sm_throughput_pct` | SM-specific throughput | Should match compute throughput |
| `dram_throughput_pct` | DRAM bandwidth usage | < 50% indicates good cache usage |

**Bottleneck Identification**:
- **Compute > Memory**: Memory-bound - optimize memory access patterns
- **Memory > Compute**: Compute-bound - increase arithmetic intensity
- **Both < 60%**: Latency-bound - check occupancy and synchronization

### Cache Performance Metrics

| Metric | Good Range | Action if Low |
|--------|-----------|---------------|
| L1 Hit Rate | > 80% | Improve data locality |
| L2 Hit Rate | > 60% | Reduce working set size |
| L1 Writeback | < 30% | Reduce write traffic |
| L2 Throughput | > 70% | Good L2 utilization |

### Pipeline Utilization

Monitor which compute pipelines are active:

| Pipeline | Purpose | Typical Usage |
|----------|---------|--------------|
| FMA | FP32 arithmetic | High for compute kernels |
| ALU | Integer ops | High for indexing-heavy |
| Tensor | Tensor cores | High for AI workloads |
| Shared | Shared memory | High for collaborative |
| FP64 | Double precision | Application-specific |

### Occupancy Analysis

Key occupancy limiters to watch:

| Limiter | Impact | Solution |
|---------|--------|----------|
| Registers | Reduces active warps | Reduce register usage |
| Shared Memory | Limits blocks | Optimize shared memory |
| Warps | Thread parallelism | Adjust block size |
| Blocks | SM utilization | Increase grid size |

## Troubleshooting

### Common Issues and Solutions

#### 1. Metrics Show "N/A%" Values

**Symptoms**: Speed of Light metrics display as "N/A%"

**Causes**:
- NCU couldn't profile the kernel (too fast or simple)
- NVTX range names don't match the expected pattern
- NCU sections not properly configured

**Solutions**:
```bash
# Ensure NVTX ranges are properly named
# Expected format: {job_id}_original and {job_id}_custom

# Verify NCU is capturing metrics
ncu --list-metrics | grep "gpu__compute_memory_throughput"

# Check if kernel is being profiled
export NCU_DEBUG=1  # Enable NCU debug logging
```

#### 2. Empty Device Metrics

**Symptoms**: `device_metrics` field is empty or missing

**Causes**:
- `ENABLE_DEVICE_METRICS` not set to `true`
- NCU not installed or not in PATH
- Insufficient permissions for GPU profiling

**Solutions**:
```bash
# Enable metrics
export ENABLE_DEVICE_METRICS=true

# Check NCU installation
which ncu || echo "NCU not found in PATH"

# Run with elevated permissions if needed
sudo -E ./run_with_ncu.sh
```

#### 3. NCU Permission Errors

**Symptoms**: "Permission denied" or "ERR_NVGPUCTRPERM"

**Solutions**:
```bash
# Option 1: Run with sudo (preserves environment)
sudo -E ./run_with_ncu.sh

# Option 2: Enable non-root profiling (system-wide)
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# Option 3: Add user to video group (Ubuntu/Debian)
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect
```

#### 4. High Profiling Overhead

**Symptoms**: Requests take much longer with metrics enabled

**Solutions**:
```bash
# Use minimal sections for faster profiling
export NCU_SECTIONS="SpeedOfLight"

# Profile only specific kernels
export NCU_KERNEL_FILTER="elementwise_add"

# Reduce replay passes
export NCU_REPLAY_MODE="kernel"  # Instead of "application"
```

## Performance Optimization Workflow

### 1. Initial Profiling

```python
import requests
import json

def profile_kernel(ref_code, custom_code):
    """Profile kernel and identify bottlenecks"""
    
    response = requests.post('http://localhost:8000/', json={
        'ref_code': ref_code,
        'custom_code': custom_code,
        'num_trials': 100
    })
    
    result = response.json()
    metrics = result['kernel_exec_result']['metadata'].get('device_metrics', {})
    
    if not metrics:
        print("No device metrics available")
        return
    
    # Extract custom kernel metrics
    custom = metrics.get('custom_device_metrics', {})
    if not custom:
        print("Custom metrics not available")
        return
        
    sol = custom.get('speed_of_light', {})
    
    # Identify bottleneck
    compute = sol.get('compute_throughput_pct', 0)
    memory = sol.get('memory_throughput_pct', 0)
    
    print(f"Compute: {compute:.1f}%, Memory: {memory:.1f}%")
    
    if memory > compute:
        print("❌ Memory-bound kernel")
        print("Recommendations:")
        print("- Improve memory coalescing")
        print("- Use shared memory for frequently accessed data")
        print("- Reduce memory transactions")
    elif compute > memory:
        print("✅ Compute-bound kernel")
        print("Recommendations:")
        print("- Good memory access patterns")
        print("- Consider increasing arithmetic intensity")
    else:
        print("⚠️ Latency-bound kernel")
        print("Recommendations:")
        print("- Check occupancy metrics")
        print("- Reduce synchronization")
        print("- Increase parallelism")
    
    # Check cache performance
    detailed = custom.get('detailed_metrics', {})
    l1_hit = detailed.get('l1_hit_rate_pct', 0)
    l2_hit = detailed.get('l2_hit_rate_pct', 0)
    
    if l1_hit < 80:
        print(f"⚠️ Low L1 hit rate: {l1_hit:.1f}%")
    if l2_hit < 60:
        print(f"⚠️ Low L2 hit rate: {l2_hit:.1f}%")
    
    return result
```

### 2. Comparative Analysis

```python
def compare_implementations(ref_code, custom_code):
    """Compare reference and custom implementations"""
    
    result = profile_kernel(ref_code, custom_code)
    metrics = result['kernel_exec_result']['metadata'].get('device_metrics', {})
    
    if not metrics:
        return
    
    orig = metrics.get('original_device_metrics', {}).get('speed_of_light', {})
    custom = metrics.get('custom_device_metrics', {}).get('speed_of_light', {})
    
    if orig and custom:
        print("\n📊 Performance Comparison:")
        print(f"  Reference - Compute: {orig.get('compute_throughput_pct', 0):.1f}%, "
              f"Memory: {orig.get('memory_throughput_pct', 0):.1f}%")
        print(f"  Custom    - Compute: {custom.get('compute_throughput_pct', 0):.1f}%, "
              f"Memory: {custom.get('memory_throughput_pct', 0):.1f}%")
        
        # Calculate improvements
        compute_diff = custom.get('compute_throughput_pct', 0) - orig.get('compute_throughput_pct', 0)
        memory_diff = custom.get('memory_throughput_pct', 0) - orig.get('memory_throughput_pct', 0)
        
        if compute_diff > 0:
            print(f"  ✅ Compute efficiency improved by {compute_diff:.1f}%")
        if memory_diff > 0:
            print(f"  ✅ Memory efficiency improved by {memory_diff:.1f}%")
```

### 3. Optimization Tracking

```python
import pandas as pd
import matplotlib.pyplot as plt

def track_optimization_progress(versions):
    """Track metrics across optimization iterations"""
    
    results = []
    
    for version_name, (ref_code, custom_code) in versions.items():
        result = profile_kernel(ref_code, custom_code)
        metrics = result['kernel_exec_result']['metadata'].get('device_metrics', {})
        
        if metrics and 'custom_device_metrics' in metrics:
            custom = metrics['custom_device_metrics']
            sol = custom.get('speed_of_light', {})
            detailed = custom.get('detailed_metrics', {})
            
            results.append({
                'version': version_name,
                'compute_pct': sol.get('compute_throughput_pct', 0),
                'memory_pct': sol.get('memory_throughput_pct', 0),
                'l1_hit_rate': detailed.get('l1_hit_rate_pct', 0),
                'l2_hit_rate': detailed.get('l2_hit_rate_pct', 0),
                'ipc': detailed.get('instructions_per_cycle', 0),
                'speedup': result['kernel_exec_result']['metadata'].get('speedup', 1.0)
            })
    
    df = pd.DataFrame(results)
    
    # Plot optimization progress
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Throughput comparison
    ax = axes[0, 0]
    x = range(len(df))
    ax.plot(x, df['compute_pct'], 'b-', label='Compute')
    ax.plot(x, df['memory_pct'], 'r-', label='Memory')
    ax.set_xticks(x)
    ax.set_xticklabels(df['version'], rotation=45)
    ax.set_ylabel('Throughput %')
    ax.set_title('GPU Throughput Evolution')
    ax.legend()
    ax.grid(True)
    
    # Cache hit rates
    ax = axes[0, 1]
    ax.plot(x, df['l1_hit_rate'], 'g-', label='L1 Cache')
    ax.plot(x, df['l2_hit_rate'], 'm-', label='L2 Cache')
    ax.set_xticks(x)
    ax.set_xticklabels(df['version'], rotation=45)
    ax.set_ylabel('Hit Rate %')
    ax.set_title('Cache Performance')
    ax.legend()
    ax.grid(True)
    
    # IPC trend
    ax = axes[1, 0]
    ax.bar(x, df['ipc'])
    ax.set_xticks(x)
    ax.set_xticklabels(df['version'], rotation=45)
    ax.set_ylabel('Instructions per Cycle')
    ax.set_title('IPC Evolution')
    ax.grid(True)
    
    # Speedup
    ax = axes[1, 1]
    ax.bar(x, df['speedup'], color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(df['version'], rotation=45)
    ax.set_ylabel('Speedup')
    ax.set_title('Performance Speedup')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Example usage
versions = {
    'v1_baseline': (ref_code_v1, custom_code_v1),
    'v2_coalesced': (ref_code_v1, custom_code_v2),
    'v3_shared_mem': (ref_code_v1, custom_code_v3),
    'v4_optimized': (ref_code_v1, custom_code_v4)
}

progress_df = track_optimization_progress(versions)
print("\nOptimization Summary:")
print(progress_df.to_string())
```

## Best Practices

### 1. Metric Collection Strategy

- **Development**: Use `SpeedOfLight` + `Occupancy` for fast iteration
- **Analysis**: Add `ComputeWorkloadAnalysis` + `MemoryWorkloadAnalysis` for deep dive
- **Validation**: Use minimal sections for regression testing

### 2. Interpreting Metrics

- **Speed of Light < 60%**: Significant optimization opportunity exists
- **Speed of Light > 80%**: Well-optimized for the hardware
- **Compute ≈ Memory**: Balanced kernel, difficult to optimize further
- **Large difference**: Clear bottleneck to address

### 3. Common Optimization Patterns

| Bottleneck | Key Metrics to Check | Common Solutions |
|------------|---------------------|------------------|
| Memory-bound | `memory_throughput_pct` > 80% | Coalescing, shared memory, compression |
| Compute-bound | `compute_throughput_pct` > 80% | Algorithm optimization, mixed precision |
| Latency-bound | Both < 60% | Increase occupancy, reduce sync |
| Cache misses | L1/L2 hit rates < 60% | Improve locality, tiling |
| Low occupancy | `waves_per_sm` < 2 | Adjust block size, reduce resources |

## Conclusion

The updated device metrics collection system provides comprehensive GPU performance insights through accurate NCU metric extraction. With six categories of metrics spanning from high-level Speed of Light analysis to detailed pipeline utilization, developers can quickly identify bottlenecks and optimize kernels effectively.

The corrected metric names and improved error handling ensure reliable profiling results, while the extensive documentation helps interpret the metrics for actionable optimization decisions.

For additional support, check the server logs, NCU documentation, or refer to the troubleshooting section above.
