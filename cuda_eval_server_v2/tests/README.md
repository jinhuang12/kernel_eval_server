# Test Suite Documentation

## Overview

This directory contains the reorganized test suite for the CUDA Evaluation Server, following a clean architecture with clear separation between unit, integration, and end-to-end tests.

## Directory Structure

```
tests/
├── unit/                          # Component-level tests (no server required)
│   ├── test_backends.py           # Compilation backend tests (includes Triton batched matmul)
│   ├── test_validators.py         # Correctness validator tests
│   ├── test_io_contract.py        # IOContract parsing tests
│   └── test_models.py             # Data model tests (RuntimeStats, DeviceMetrics, etc.)
│
├── integration/                   # Server integration tests
│   ├── test_endpoints.py          # API endpoint tests (includes percentile validation)
│   ├── test_kernels.py            # Kernel-specific tests
│   ├── test_failure.py            # Error handling tests (includes Triton error reporting)
│   ├── test_multi_kernel.py       # Multi-kernel support tests (Triton+PyTorch+CUDA)
│   └── test_triton_metadata.py    # Triton kernel selection with metadata
│
├── e2e/                           # End-to-end workflow tests
│   └── test_workflows.py          # Complete evaluation workflows
│
├── mcp/                           # MCP server tests
│   ├── test_mcp_tools.py          # MCP tool tests
│   ├── test_mcp_integration.py    # MCP integration tests
│   └── test_mcp_error_handling.py # MCP error handling
│
├── fixtures/                      # Shared test utilities
│   ├── kernels.py                 # Sample kernel library
│   ├── factories.py               # Request/response factories
│   ├── validators.py              # Response validation helpers
│   └── test_data_loader.py        # Test data loading from JSON
│
└── conftest.py                    # Pytest configuration
```

## Running Tests

### Prerequisites

1. Activate conda environment:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
```

2. For integration tests, ensure server is running:
```bash
./docker-build.sh && ./docker-run.sh --gpu all
```

### Test Commands

Run all tests:
```bash
python3 -m pytest tests/
```

Run specific test categories:
```bash
# Unit tests only
python3 -m pytest tests/unit/

# Integration tests only
python3 -m pytest tests/integration/

# E2E tests only
python3 -m pytest tests/e2e/
```

Run with markers:
```bash
# GPU-required tests
python3 -m pytest -m gpu

# Triton-specific tests
python3 -m pytest -m triton

# Skip slow tests
python3 -m pytest -m "not slow"
```

## Test Organization

### Unit Tests (`tests/unit/`)

- **Purpose**: Test individual components in isolation
- **Requirements**: No server needed, minimal dependencies
- **Coverage**: 
  - Compilation backends (Torch, Triton, CUDA)
  - Validation logic
  - Data models and parsing

### Integration Tests (`tests/integration/`)

- **Purpose**: Test API endpoints and server interactions
- **Requirements**: Running server instance
- **Coverage**:
  - All API endpoints (/evaluate, /compare, /health)
  - Different kernel types
  - Error handling and edge cases
  - Client error (400) vs server error (500) responses

### E2E Tests (`tests/e2e/`)

- **Purpose**: Test complete workflows and real-world scenarios
- **Requirements**: Running server with full capabilities
- **Coverage**:
  - Multi-step optimization workflows
  - Performance benchmarking scenarios
  - Kernel fusion and layer optimization
  - Stress testing

## Shared Fixtures (`tests/fixtures/`)

### KernelLibrary (`kernels.py`)
Provides sample kernels for testing:
- `torch_add()`, `torch_matmul()`, `torch_gelu()` - PyTorch kernels
- `torch_cuda_add()` - PyTorch with embedded CUDA
- `triton_add()`, `triton_matmul()` - Triton kernels
- Error kernels for failure testing

### RequestFactory (`factories.py`)
Creates properly formatted API requests:
- `create_evaluate_request()` - For /evaluate endpoint
- `create_compare_request()` - For /compare endpoint
- `create_invalid_request()` - For error testing

### ResponseValidator (`validators.py`)
Validates API responses:
- Status code validation
- Response format checking
- Performance metrics validation
- Error response validation

## New Test Files (Oct 2025 Refactoring)

### Unit Tests
- **test_models.py**: Tests for data models (RuntimeStats, DeviceMetrics, KernelMetadata, KernelExecutionResult, response models)
  - Model creation and serialization
  - to_dict() and from_dict() conversions
  - JSON serialization verification
  - Model integration tests

### Integration Tests
- **test_multi_kernel.py**: Multi-kernel support tests
  - Simple Triton+PyTorch sequences
  - Complex 2 Triton + 2 CUDA kernel sequences
  - Error handling (missing entry points)
  - Performance testing with varying trial counts

- **test_triton_metadata.py**: Triton kernel selection and metadata
  - Kernel selection by name from multi-kernel sources
  - Metadata preservation through API endpoints
  - Default kernel behavior
  - Invalid kernel name error handling
  - Dict-based metadata support

### Enhanced Tests
- **test_endpoints.py**: Added runtime statistics validation
  - Percentile validation (p95, p99)
  - Statistical consistency checks (min ≤ mean ≤ max, etc.)
  - Complete field verification for runtime_stats and ref_runtime

- **test_failure.py**: Added Triton error reporting
  - Full stack trace visibility for Triton constraint errors
  - Matrix dimension constraint testing (tl.dot M < 16 error)

- **test_backends.py**: Added complex Triton test
  - Batched matrix multiplication with 3D tensors
  - Launch configuration with multi-dimensional grids
  - Stride parameter handling

## Key Improvements

1. **60% Code Reduction**: From ~9000 to ~3500 lines through deduplication
2. **Clear Organization**: Tests grouped by purpose and requirements
3. **Shared Utilities**: Common fixtures eliminate boilerplate
4. **Flexible Testing**: Parametrized tests for multiple scenarios
5. **Better Maintainability**: Single source of truth for test data
6. **Zero Root-Level Tests**: All tests properly organized in tests/ directory

## Migration from Old Tests

### Root-Level Test Files (Migrated & Deleted)
The following root-level test files have been refactored and deleted:
- `test_multi_kernel.py` → `tests/integration/test_multi_kernel.py`
- `test_triton.py` → `tests/unit/test_backends.py` (batched matmul test)
- `test_triton_multi_kernel.py` → `tests/integration/test_triton_metadata.py`
- `test_typed_models.py` → `tests/unit/test_models.py`
- `test_mcp_server.py` → `tests/mcp/` (existing tests cover functionality)
- `test_runtime_stats.py` → `tests/integration/test_endpoints.py` (percentile tests)
- `test_percentiles.py` → **DELETED** (duplicate of test_runtime_stats.py)
- `test_error_reporting.py` → `tests/integration/test_failure.py` (Triton error tests)
- `test_metadata_compare.py` → `tests/integration/test_triton_metadata.py`

### old_tests/ Directory Files (Archived)
Old test files in old_tests/ have been consolidated:
- `test_server_*.py` → `tests/integration/`
- `test_*_backend.py` → `tests/unit/test_backends.py`
- `test_correctness_validator.py` → `tests/unit/test_validators.py`
- `test_io_contract.py` → `tests/unit/test_io_contract.py`
- `test_server_comprehensive.py` → `tests/e2e/test_workflows.py`

See `old_tests/DEPRECATED.md` for complete migration details.

## Known Issues

1. **API Format Mismatch**: The server returns different response formats for different endpoints. The validators handle both formats.

2. **Kernel Requirements**: Torch kernels may require specific Model/get_inputs patterns or IOContract specifications.

3. **Environment Setup**: Tests require proper conda environment activation on EC2 instance.

## Contributing

When adding new tests:
1. Place in appropriate directory (unit/integration/e2e)
2. Use shared fixtures from `tests/fixtures/`
3. Add appropriate pytest markers (@pytest.mark.unit, etc.)
4. Follow existing patterns for consistency