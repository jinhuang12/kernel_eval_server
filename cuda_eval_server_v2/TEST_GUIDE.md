# Comprehensive Testing Guide for CUDA Evaluation Server V2

## Overview

The refactored test suite provides a modular, maintainable, and efficient testing framework for the CUDA Evaluation Server. The new implementation reduces code from ~1764 lines to ~950 lines while improving test coverage and adding new capabilities.

## Test Architecture

### Core Components

1. **test_utils.py** (~400 lines)
   - `TestCaseProvider`: Unified test case loading from KernelBench JSON
   - `ServerClient`: HTTP client for server communication
   - `ResultAnalyzer`: Extract and analyze server responses
   - `MetricsCollector`: Collect and aggregate test metrics
   - `TestReporter`: Generate and display test reports

2. **test_server_comprehensive.py** (~550 lines)
   - `FunctionalTests`: Basic functionality and error handling
   - `PerformanceTests`: Load testing and performance validation
   - `IntegrationTests`: Multi-kernel and stress testing
   - `ComprehensiveTestSuite`: Test orchestration

## Test Categories

### 1. Functional Tests
- **Basic Functionality**: Health check, stats endpoint, simple evaluation
- **Validation Accuracy**: Compare actual vs expected results
- **Error Handling**: Malformed requests, timeouts, syntax errors

### 2. Performance Tests
- **Concurrent Load**: Test parallel request handling
- **GPU Resource Management**: Verify proper resource acquisition/release
- **Compilation Cache**: Measure cache effectiveness
- **Sustained Load**: Long-running stability tests

### 3. Integration Tests
- **Specific Case Testing**: Debug individual test cases by index
- **Multi-Kernel Types**: Test TORCH, TORCH_CUDA, TRITON kernels
- **Stress Patterns**: Various load patterns scaled to GPU count

## Usage Examples

### Basic Testing
```bash
# Quick test (5 cases, 3 concurrent requests)
python test_server_comprehensive.py --mode quick

# Basic functionality only
python test_server_comprehensive.py --mode basic

# Full validation suite
python test_server_comprehensive.py --mode validation --batch-size 30
```

### Performance Testing
```bash
# Load testing with custom concurrency
python test_server_comprehensive.py --mode load --concurrent 10

# Performance suite with sustained load
python test_server_comprehensive.py --mode performance

# Cache effectiveness testing
python test_server_comprehensive.py --mode performance --trials 100
```

### Debug and Analysis
```bash
# Test specific case by index
python test_server_comprehensive.py --index 396 --trials 10

# Test with device metrics (NCU profiling)
python test_server_comprehensive.py --index 396 --device-metrics

# Verbose output with detailed metrics
python test_server_comprehensive.py --mode comprehensive --verbose

# Save results to JSON
python test_server_comprehensive.py --mode comprehensive --save-report results.json
```

### Comprehensive Testing
```bash
# Full test suite (recommended for CI/CD)
python test_server_comprehensive.py --mode comprehensive

# Custom configuration
python test_server_comprehensive.py \
    --server http://localhost:8000 \
    --mode comprehensive \
    --trials 20 \
    --batch-size 50 \
    --concurrent 8 \
    --timeout 180 \
    --verbose
```

### Multi-Kernel Testing (NEW)
```bash
# Test all kernel types
python test_server_comprehensive.py --mode comprehensive --kernel-types all

# Test only Triton kernels
python test_server_comprehensive.py --mode triton --kernel-types triton --iocontract

# Mixed TORCH_CUDA and Triton testing
python test_server_comprehensive.py --mode comprehensive --kernel-types torch_cuda triton

# Triton with optimization levels
python test_server_comprehensive.py --mode triton --kernel-types triton --triton-opt both

# Compare same operation across kernel types
python test_server_comprehensive.py --mode triton --kernel-types all --trials 100
```

## Test Modes

| Mode | Description | Duration | Use Case |
|------|-------------|----------|----------|
| `basic` | Basic server functionality | ~10s | Quick sanity check |
| `quick` | Small validation batch | ~30s | Pre-commit testing |
| `validation` | Full validation accuracy | ~2m | Correctness verification |
| `load` | Performance and load tests | ~5m | Stress testing |
| `performance` | All performance tests | ~8m | Performance validation |
| `comprehensive` | Complete test suite | ~15m | Full CI/CD pipeline |
| `triton` | Triton kernel testing | ~5m | Triton-specific validation |
| `debug` | Single test case analysis | ~10s | Debugging specific issues |

## Key Improvements Over Original

### Code Reduction (45%)
- **Before**: 1764 lines with significant redundancy
- **After**: ~950 lines with modular structure

### Enhanced Capabilities
1. **Unified Test Provider**: Single source of truth for test cases
2. **Parameterized Testing**: Configurable test execution
3. **Better Error Handling**: Graceful failure scenarios
4. **Performance Baselines**: Track regression over time
5. **Resource Leak Detection**: GPU management validation
6. **Cache Verification**: Compilation cache effectiveness

### NEW: Unified Multi-Kernel Testing
1. **Supports All Kernel Types**: TORCH, TORCH_CUDA, TRITON, CUDA
2. **IOContract Testing**: JSON-based test specifications from test_data/triton/
3. **Mixed Kernel Batches**: Interleave different kernel types in single test run
4. **Kernel Comparison**: Compare same operation across different implementations
5. **Triton Optimizations**: Test naive vs optimized Triton implementations
6. **Unified Reporting**: Consolidated metrics across all kernel types

### Improved Maintainability
- Modular test categories
- Reusable utilities
- Clear separation of concerns
- Comprehensive documentation

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--server` | http://localhost:8000 | Server URL |
| `--mode` | comprehensive | Test execution mode |
| `--trials` | 10 | Trials per kernel evaluation |
| `--batch-size` | 20 | Validation test batch size |
| `--concurrent` | 5 | Concurrent request count |
| `--timeout` | 120 | Request timeout (seconds) |
| `--verbose` | False | Enable verbose output |
| `--save-report` | None | Save results to JSON file |
| `--device-metrics` | False | Enable NCU device metrics collection |
| `--kernel-types` | torch_cuda | Kernel types to test (torch, torch_cuda, triton, all) |
| `--triton-opt` | naive | Triton optimization level (naive, optimized, both) |
| `--iocontract` | False | Enable IOContract-based Triton testing |

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run comprehensive tests
  run: |
    python test_server_comprehensive.py \
      --mode comprehensive \
      --save-report test-results.json
    
- name: Upload test results
  uses: actions/upload-artifact@v2
  with:
    name: test-results
    path: test-results.json
```

### Jenkins Pipeline Example
```groovy
stage('Test') {
    steps {
        sh '''
            python test_server_comprehensive.py \
              --mode validation \
              --batch-size 50
        '''
    }
}
```

## Troubleshooting

### Common Issues

1. **No test cases loaded**
   - Ensure `kernelbench_evaluated_20250808_212140.json` exists
   - Check file path in TestConfig

2. **Server connection failed**
   - Verify server is running: `python main.py`
   - Check server URL and port

3. **GPU resource errors**
   - Ensure GPUs are available
   - Check CUDA installation

4. **Timeout errors**
   - Increase timeout: `--timeout 300`
   - Reduce batch size: `--batch-size 10`

## Performance Metrics

The test suite tracks various performance metrics:

- **Response Times**: Mean, median, min, max, stdev
- **GPU Utilization**: Distribution across available GPUs
- **Cache Effectiveness**: Hit rate and speedup
- **Throughput**: Requests per second
- **Success Rate**: Percentage of successful tests
- **Validation Accuracy**: Expected vs actual results

## Future Enhancements

1. **Memory Profiling**: Track memory usage over time
2. **Regression Detection**: Compare against historical baselines
3. **Distributed Testing**: Support for multi-node testing
4. **Custom Test Scenarios**: User-defined test patterns
5. **Real-time Monitoring**: Live dashboard during tests