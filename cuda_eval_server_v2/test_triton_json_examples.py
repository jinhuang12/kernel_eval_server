import json
import os
import pytest
import torch

from shared.models import KernelCode, KernelType
from validation.correctness_validator import CorrectnessValidator

# Attempt to import backends; tests will skip if missing dependencies
try:
    from compilation.triton import TritonCompilationBackend
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

try:
    from compilation.torch import TorchCompilationBackend
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "test_data", "triton_examples")
EXAMPLE_FILES = [
    "torch_triton_add.json",
    "triton_matmul_optimized.json",
]

@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_triton_examples_from_json(filename):
    if not torch.cuda.is_available() or not HAVE_TRITON or not HAVE_TORCH:
        pytest.skip("Required dependencies or CUDA not available")

    path = os.path.join(EXAMPLES_DIR, filename)
    with open(path, "r") as f:
        data = json.load(f)

    ref_kernel = KernelCode.from_dict(data["ref_kernel"])
    custom_kernel = KernelCode.from_dict(data["custom_kernel"])

    backends = {
        KernelType.TRITON: TritonCompilationBackend(),
        KernelType.TORCH: TorchCompilationBackend(),
    }

    ref_exec = backends[ref_kernel.kernel_type].compile(ref_kernel, gpu_id=0)
    custom_exec = backends[custom_kernel.kernel_type].compile(custom_kernel, gpu_id=0)

    validator = CorrectnessValidator()
    result = validator.validate_correctness(ref_exec, custom_exec, device=torch.device("cuda:0"), num_correct_trials=1)
    assert result.is_correct, result.error
