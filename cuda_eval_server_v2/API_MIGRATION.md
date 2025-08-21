# API Migration Guide

## Overview

The CUDA Evaluation Server V2 maintains backward compatibility with V1 while introducing a more structured API using typed kernel objects. This guide helps you migrate from the old to the new API format.

## API Format Comparison

### Old API (V1)

```json
{
  "ref_code": "import torch...",
  "custom_code": "import torch...",
  "num_trials": 100,
  "timeout": 120
}
```

### New API (V2)

```json
{
  "ref_kernel": {
    "source_code": "import torch...",
    "kernel_type": "torch"
  },
  "custom_kernel": {
    "source_code": "import torch...",
    "kernel_type": "torch_cuda"
  },
  "num_trials": 100,
  "timeout": 120
}
```

## Key Changes

### 1. Kernel Objects Instead of Strings

**Old**: Plain string fields `ref_code` and `custom_code`

**New**: `KernelCode` objects with:
- `source_code`: The actual code (string)
- `kernel_type`: Type enum (torch, torch_cuda, cuda, triton)
- `metadata`: Optional additional information

### 2. Explicit Kernel Types

**Old**: Server guessed kernel type from code content

**New**: Explicit declaration of kernel type enables:
- Correct compilation strategy selection
- Better error messages
- Support for multiple kernel types

### 3. Field Name Changes

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `ref_code` | `ref_kernel.source_code` | Now part of kernel object |
| `custom_code` | `custom_kernel.source_code` | Now part of kernel object |
| N/A | `ref_kernel.kernel_type` | New required field |
| N/A | `custom_kernel.kernel_type` | New required field |

## Migration Examples

### Example 1: Simple Migration

**Old Request**:
```python
import requests

response = requests.post('http://localhost:8000/', json={
    'ref_code': reference_model_code,
    'custom_code': custom_kernel_code,
    'num_trials': 100
})
```

**New Request**:
```python
import requests

response = requests.post('http://localhost:8000/', json={
    'ref_kernel': {
        'source_code': reference_model_code,
        'kernel_type': 'torch'
    },
    'custom_kernel': {
        'source_code': custom_kernel_code,
        'kernel_type': 'torch_cuda'
    },
    'num_trials': 100
})
```

### Example 2: With Timeout

**Old**:
```python
request_data = {
    'ref_code': ref_code,
    'custom_code': custom_code,
    'num_trials': 50,
    'timeout': 180
}
```

**New**:
```python
request_data = {
    'ref_kernel': {
        'source_code': ref_code,
        'kernel_type': 'torch'
    },
    'custom_kernel': {
        'source_code': custom_code,
        'kernel_type': 'torch_cuda'
    },
    'num_trials': 50,
    'timeout': 180
}
```

## Kernel Type Selection Guide

### When to use `torch`
- Pure PyTorch models
- Reference implementations
- No CUDA code

Example:
```python
class Model(nn.Module):
    def forward(self, x):
        return x + 1
```

### When to use `torch_cuda`
- PyTorch models with embedded CUDA
- Uses `torch.utils.cpp_extension.load_inline`
- Custom CUDA kernels wrapped in PyTorch

Example:
```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void my_kernel(...) { ... }
torch::Tensor my_function(...) { ... }
"""

module = load_inline(name="my_module", ...)

class ModelNew(nn.Module):
    def forward(self, x):
        return module.my_function(x)
```

### When to use `cuda` (Future)
- Raw CUDA kernels
- Direct CUDA code without PyTorch wrapper

### When to use `triton` (Future)
- Triton kernel implementations
- Triton-specific optimizations

## Backward Compatibility

The server automatically detects old format requests and converts them:

```python
# In app.py
if "ref_code" in request and "custom_code" in request:
    # Old format - convert to new format
    evaluation_request = EvaluationRequest(
        ref_kernel=KernelCode(
            source_code=request["ref_code"],
            kernel_type=KernelType.TORCH
        ),
        custom_kernel=KernelCode(
            source_code=request["custom_code"],
            kernel_type=KernelType.TORCH_CUDA
        ),
        num_trials=request.get("num_trials", 100),
        timeout=request.get("timeout", 120)
    )
```

**Note**: When using backward compatibility:
- Reference code is assumed to be `torch` type
- Custom code is assumed to be `torch_cuda` type
- For other kernel types, use the new API format

## Response Format

The response format remains largely unchanged:

```json
{
  "status": "success",
  "job_id": "uuid-here",
  "kernel_exec_result": {
    "compiled": true,
    "correctness": true,
    "runtime": 1.234,
    "runtime_stats": {...},
    "metadata": {
      "gpu_id": 0,
      "device_metrics": {...}
    }
  },
  "ref_runtime": {...},
  "pod_name": "hostname",
  "pod_ip": "ip-address"
}
```

## Migration Checklist

- [ ] Identify kernel types for your code
- [ ] Update request format to use kernel objects
- [ ] Specify correct kernel_type for each kernel
- [ ] Update error handling for new error messages
- [ ] Test with both old and new formats
- [ ] Update documentation and examples

## Common Migration Issues

### Issue 1: Missing kernel_type

**Error**: `ValueError: No compilation strategy available for kernel type: None`

**Solution**: Ensure you specify kernel_type for each kernel

### Issue 2: Wrong kernel_type

**Error**: Compilation fails with type-specific errors

**Solution**: Verify kernel_type matches actual code:
- Pure PyTorch → `torch`
- PyTorch with CUDA → `torch_cuda`

### Issue 3: Using old field names with new structure

**Error**: `KeyError: 'source_code'`

**Solution**: Use correct nested structure:
```python
# Wrong
{'ref_kernel': my_code}

# Correct
{'ref_kernel': {'source_code': my_code, 'kernel_type': 'torch'}}
```

## Benefits of Migration

1. **Explicit Types**: No ambiguity about kernel implementation
2. **Better Error Messages**: Type-specific error handling
3. **Future-Proof**: Support for new kernel types
4. **Improved Performance**: Correct strategy selection
5. **Enhanced Debugging**: Type information in logs

## Gradual Migration Strategy

1. **Phase 1**: Test new API format in development
2. **Phase 2**: Update new code to use new format
3. **Phase 3**: Gradually migrate existing code
4. **Phase 4**: Deprecate old format usage

## Support

- Use backward compatibility during transition
- Monitor deprecation warnings in logs
- Test thoroughly with both formats
- Report issues with migration

## Conclusion

While the old API format is still supported, migrating to the new format provides better type safety, clearer intent, and access to new features. The structured kernel objects make the API more maintainable and extensible for future enhancements.
