# tests/test_correctness_validator.py
import math
import pytest
import torch

# ---- import your primitives -------------------------------------------------
from shared.executable_kernels import BaseExecutableKernel
from shared.models import (
    ArgSpec, TensorSpec, IOContract, LaunchConfig, LaunchDim,
    KernelCode, KernelType, ValidationResult
)
from validation.correctness_validator import CorrectnessValidator

# Optional Triton imports (tests are skipped if not found)
try:
    import triton  # type: ignore
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0") if CUDA_AVAILABLE else torch.device("cpu")

# =============================================================================
# Dummy kernels (subclasses of BaseExecutableKernel) for fast + deterministic tests
# =============================================================================

class DummyAddKernel(BaseExecutableKernel):
    """
    Simple kernel: returns x + y. Uses _default_inputs if none passed.
    """
    def __init__(self, device: torch.device, default_inputs=True):
        super().__init__(kernel_type=KernelType.TORCH, device=device, io_contract=None)
        if default_inputs:
            g = torch.Generator(device=device); g.manual_seed(123)
            x = torch.randn(8, device=device, generator=g)
            y = torch.randn(8, device=device, generator=g)
            self._default_inputs = (x, y)

    def _initialize_kernel(self):
        pass

    def _execute_impl(self, *inputs):
        if not inputs:
            inputs = self._default_inputs
        x, y = inputs
        return x + y


class DummyAddKernelFromSpecs(BaseExecutableKernel):
    """
    Same as DummyAddKernel but with no _default_inputs.
    Validates validator's ability to materialize inputs from IOContract.input_specs.
    """
    def __init__(self, device: torch.device):
        io = IOContract(args=[
            ArgSpec(name="x", type="tensor", role="input",
                    tensor_spec=TensorSpec(shape=[8], dtype="float32")),
            ArgSpec(name="y", type="tensor", role="input",
                    tensor_spec=TensorSpec(shape=[8], dtype="float32")),
        ])
        super().__init__(kernel_type=KernelType.TORCH, device=device, io_contract=io)

    def _initialize_kernel(self): pass

    def _execute_impl(self, *inputs):
        assert len(inputs) == 2, "expects two tensors"
        x, y = inputs
        return x + y


class DummyAddKernelWithEpsilon(BaseExecutableKernel):
    """
    Returns x + y + eps, for tolerance tests.
    """
    def __init__(self, device: torch.device, eps: float):
        super().__init__(kernel_type=KernelType.TORCH, device=device, io_contract=None)
        self.eps = float(eps)
        g = torch.Generator(device=device); g.manual_seed(123)
        x = torch.randn(8, device=device, generator=g)
        y = torch.randn(8, device=device, generator=g)
        self._default_inputs = (x, y)

    def _initialize_kernel(self): pass

    def _execute_impl(self, *inputs):
        if not inputs:
            inputs = self._default_inputs
        x, y = inputs
        return x + y + self.eps


class DummyMultiOutKernel(BaseExecutableKernel):
    """
    Returns (x, x + y) to test multi-output comparison.
    """
    def __init__(self, device: torch.device):
        super().__init__(kernel_type=KernelType.TORCH, device=device, io_contract=None)
        g = torch.Generator(device=device); g.manual_seed(321)
        x = torch.randn(4, device=device, generator=g)
        y = torch.randn(4, device=device, generator=g)
        self._default_inputs = (x, y)

    def _initialize_kernel(self): pass

    def _execute_impl(self, *inputs):
        if not inputs:
            inputs = self._default_inputs
        x, y = inputs
        return (x, x + y)


class DummyWrongShapeKernel(BaseExecutableKernel):
    """
    Returns a tensor with a different shape to trigger shape mismatch.
    """
    def __init__(self, device: torch.device):
        super().__init__(kernel_type=KernelType.TORCH, device=device, io_contract=None)
        g = torch.Generator(device=device); g.manual_seed(123)
        x = torch.randn(8, device=device, generator=g)
        y = torch.randn(8, device=device, generator=g)
        self._default_inputs = (x, y)

    def _initialize_kernel(self): pass

    def _execute_impl(self, *inputs):
        if not inputs:
            inputs = self._default_inputs
        # wrong shape on purpose
        return torch.zeros(9, device=self.device)


class DummyWarmupErrorKernel(BaseExecutableKernel):
    """
    Raises on first call (during warmup) to test error propagation.
    """
    def __init__(self, device: torch.device):
        super().__init__(kernel_type=KernelType.TORCH, device=device, io_contract=None)
        g = torch.Generator(device=device); g.manual_seed(999)
        x = torch.randn(8, device=device, generator=g)
        y = torch.randn(8, device=device, generator=g)
        self._default_inputs = (x, y)
        self._first = True

    def _initialize_kernel(self): pass

    def _execute_impl(self, *inputs):
        if self._first:
            self._first = False
            raise RuntimeError("warmup-fail")
        x, y = inputs if inputs else self._default_inputs
        return x + y

# =============================================================================
# Triton kernel (optional test) — auto-skipped if Triton/CUDA unavailable
# =============================================================================

TRITON_ADD_SRC = '''
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)
'''

@pytest.fixture(scope="module")
def triton_executable_if_available():
    if not CUDA_AVAILABLE or not HAVE_TRITON:
        pytest.skip("CUDA or Triton not available")

    # Import your TritonExecutableKernel (or use your backend to compile)
    # Adjust import path to your project layout:
    from compilation.triton import TritonExecutableKernel  # type: ignore

    io = IOContract(
        args=[
            ArgSpec(name="x_ptr", type="tensor", role="input",
                    tensor_spec=TensorSpec(shape=[2048], dtype="float32")),
            ArgSpec(name="y_ptr", type="tensor", role="input",
                    tensor_spec=TensorSpec(shape=[2048], dtype="float32")),
            ArgSpec(name="out_ptr", type="tensor", role="output",
                    tensor_spec=TensorSpec(shape=[2048], dtype="float32")),
            ArgSpec(name="n_elements", type="int", value=2048),
            ArgSpec(name="BLOCK_SIZE", type="int", value=256, is_meta=True),
        ],
        outputs=[TensorSpec(shape=[2048], dtype="float32")],
        launch=LaunchConfig(grid=LaunchDim(x=8), num_warps=4)
    )
    kc = KernelCode(source_code=TRITON_ADD_SRC, kernel_type=KernelType.TRITON, io=io, metadata={"name": "add_kernel"})
    exe = TritonExecutableKernel(kc, device=DEVICE, kernel_name="add_kernel")
    return exe

# =============================================================================
# Tests
# =============================================================================

def test_torch_vs_torch_identical():
    ref = DummyAddKernel(device=DEVICE)
    cand = DummyAddKernel(device=DEVICE)
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=3)
    assert isinstance(res, ValidationResult)
    assert res.is_correct
    assert res.trials_passed == 3
    assert res.total_trials == 3

def test_torch_vs_torch_from_specs():
    ref = DummyAddKernelFromSpecs(device=DEVICE)
    cand = DummyAddKernelFromSpecs(device=DEVICE)
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=2)
    assert res.is_correct
    assert res.trials_passed == 2

def test_tolerance_small_eps_passes():
    ref = DummyAddKernel(device=DEVICE)
    cand = DummyAddKernelWithEpsilon(device=DEVICE, eps=1e-3)  # well below 1e-2 tol
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=2)
    assert res.is_correct

def test_tolerance_large_eps_fails():
    ref = DummyAddKernel(device=DEVICE)
    cand = DummyAddKernelWithEpsilon(device=DEVICE, eps=5e-2)  # above 1e-2 tol
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=1)
    assert not res.is_correct
    assert res.trials_passed == 0
    assert res.total_trials == 1
    assert res.max_difference is not None

def test_multi_output_pairwise_compare():
    class DummyMultiOutCand(DummyMultiOutKernel):
        def _execute_impl(self, *inputs):
            if not inputs:
                inputs = self._default_inputs
            x, y = inputs
            return (x, x + y)  # identical to ref

    ref = DummyMultiOutKernel(device=DEVICE)
    cand = DummyMultiOutCand(device=DEVICE)
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=2)
    assert res.is_correct
    assert res.trials_passed == 2

def test_shape_mismatch_fails():
    ref = DummyAddKernel(device=DEVICE)
    cand = DummyWrongShapeKernel(device=DEVICE)
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=1)
    assert not res.is_correct
    assert "Shape mismatch" in (res.error or "")

def test_warmup_failure_reports_error():
    ref = DummyAddKernel(device=DEVICE)
    cand = DummyWarmupErrorKernel(device=DEVICE)
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=1)
    assert not res.is_correct
    assert "Warmup failed" in (res.error or "") or "warmup-fail" in (res.error or "")

def test_missing_inputs_reports_error():
    class DummyNoInputs(BaseExecutableKernel):
        def _initialize_kernel(self): pass
        def _execute_impl(self, *inputs):
            return torch.zeros(1, device=self.device)

    ref = DummyNoInputs(KernelType.TORCH, device=DEVICE)
    cand = DummyNoInputs(KernelType.TORCH, device=DEVICE)
    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=1)
    assert not res.is_correct
    assert "determine inputs" in (res.error or "").lower()

@pytest.mark.skipif(not (CUDA_AVAILABLE and HAVE_TRITON), reason="CUDA/Triton required")
def test_torch_vs_triton_add(triton_executable_if_available):
    # Reference torch add kernel with DEFAULT INPUTS OF SIZE 2048 to match Triton IOContract
    ref = DummyAddKernel(device=DEVICE, default_inputs=False)
    g = torch.Generator(device=DEVICE); g.manual_seed(123)
    x = torch.randn(2048, device=DEVICE, generator=g)
    y = torch.randn(2048, device=DEVICE, generator=g)
    ref._default_inputs = (x, y)

    # Custom = Triton add using IOContract path (expects 2048-length vectors)
    cand = triton_executable_if_available

    val = CorrectnessValidator()
    res = val.validate_correctness(ref, cand, DEVICE, num_correct_trials=2)

    assert res.is_correct, f"Expected correctness vs Triton add, got: {res}"
    assert res.trials_passed == 2
