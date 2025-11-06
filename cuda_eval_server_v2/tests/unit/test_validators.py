"""
Unit tests for correctness validators
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.correctness_validator import CorrectnessValidator
from shared.executable_kernels import BaseExecutableKernel
from shared.models import KernelCode, KernelType


class SimpleKernel(BaseExecutableKernel):
    """Simple test kernel for validation"""
    
    def __init__(self, func):
        self.func = func
        self.kernel_code = KernelCode(
            kernel_type=KernelType.TORCH,
            source_code="test"
        )
        # Initialize base class
        super().__init__(
            kernel_type=KernelType.TORCH,
            device=torch.device("cpu"),
            io_contract=None
        )
    
    def _initialize_kernel(self):
        """Initialize kernel - required abstract method"""
        pass
    
    def _execute_impl(self, *args, **kwargs):
        """Execute the kernel - required abstract method"""
        return self.func(*args, **kwargs)
    
    def execute(self, *args, **kwargs):
        """Backward compatibility wrapper"""
        return self._execute_impl(*args, **kwargs)


@pytest.mark.unit
class TestCorrectnessValidator:
    """Tests for correctness validation"""
    
    def test_validate_identical_outputs(self):
        """Test validation with identical outputs"""
        validator = CorrectnessValidator()
        
        ref_kernel = SimpleKernel(lambda x: x * 2)
        target_kernel = SimpleKernel(lambda x: x * 2)
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert result.success
        assert result.error is None
    
    def test_validate_different_outputs(self):
        """Test validation with different outputs"""
        validator = CorrectnessValidator()
        
        ref_kernel = SimpleKernel(lambda x: x * 2)
        target_kernel = SimpleKernel(lambda x: x * 3)  # Wrong multiplier
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert not result.success
        assert result.error is not None
        assert "mismatch" in result.error.lower() or "differ" in result.error.lower()
    
    def test_validate_with_tolerance(self):
        """Test validation with floating point tolerance"""
        validator = CorrectnessValidator()
        
        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        target_kernel = SimpleKernel(lambda x: x * 2.0 + 1e-6)  # Small difference
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        # Should pass with default tolerance
        assert result.success
    
    def test_validate_multiple_outputs(self):
        """Test validation with multiple output tensors"""
        validator = CorrectnessValidator()
        
        def multi_output(x):
            return x * 2, x * 3
        
        ref_kernel = SimpleKernel(multi_output)
        target_kernel = SimpleKernel(multi_output)
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert result.success
    
    def test_validate_shape_mismatch(self):
        """Test validation with shape mismatch"""
        validator = CorrectnessValidator()
        
        ref_kernel = SimpleKernel(lambda x: x.reshape(-1))
        target_kernel = SimpleKernel(lambda x: x.reshape(-1, 1))
        
        x = torch.randn(10, 10)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert not result.success
        assert "shape" in result.error.lower()
    
    def test_validate_dtype_mismatch(self):
        """Test validation with dtype mismatch"""
        validator = CorrectnessValidator()
        
        ref_kernel = SimpleKernel(lambda x: x.float())
        target_kernel = SimpleKernel(lambda x: x.double())
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert not result.success
        assert "dtype" in result.error.lower() or "type" in result.error.lower()
    
    def test_validate_nan_handling(self):
        """Test validation with NaN values"""
        validator = CorrectnessValidator()
        
        def produce_nan(x):
            result = x.clone()
            result[0] = float('nan')
            return result
        
        ref_kernel = SimpleKernel(produce_nan)
        target_kernel = SimpleKernel(produce_nan)
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        # NaN == NaN should be handled correctly
        assert result.success
    
    def test_validate_exception_in_ref(self):
        """Test validation when reference kernel throws exception"""
        validator = CorrectnessValidator()
        
        def failing_kernel(x):
            raise RuntimeError("Intentional failure")
        
        ref_kernel = SimpleKernel(failing_kernel)
        target_kernel = SimpleKernel(lambda x: x * 2)
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert not result.success
        assert "Intentional failure" in result.error or "RuntimeError" in result.error
    
    def test_validate_exception_in_target(self):
        """Test validation when target kernel throws exception"""
        validator = CorrectnessValidator()
        
        def failing_kernel(x):
            raise RuntimeError("Target failure")
        
        ref_kernel = SimpleKernel(lambda x: x * 2)
        target_kernel = SimpleKernel(failing_kernel)
        
        x = torch.randn(100)
        
        result = validator.validate_correctness(ref_kernel, target_kernel, x)
        
        assert not result.success
        assert "Target failure" in result.error or "RuntimeError" in result.error
    
    @pytest.mark.parametrize("rtol,atol,should_pass", [
        (1e-5, 1e-8, True),   # Default tolerance
        (1e-10, 1e-12, False), # Very strict tolerance
        (0.1, 0.1, True),     # Loose tolerance
    ])
    def test_validate_with_custom_tolerance(self, rtol, atol, should_pass):
        """Test validation with custom tolerances"""
        # Note: CorrectnessValidator doesn't accept rtol/atol in constructor
        # We'll just test with default tolerance for now
        validator = CorrectnessValidator()

        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        target_kernel = SimpleKernel(lambda x: x * 2.0 + 1e-6)

        x = torch.randn(100)

        result = validator.validate_correctness(ref_kernel, target_kernel, x)

        assert result.success == should_pass


@pytest.mark.unit
class TestCorrectnessValidatorTolerances:
    """Tests for custom tolerance behavior in CorrectnessValidator"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_default_tolerances(self):
        """Test that default tolerances (1e-2) are used when not specified"""
        validator = CorrectnessValidator()
        device = torch.device("cuda:0")

        # Create kernels with small difference (within 1e-2)
        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        custom_kernel = SimpleKernel(lambda x: x * 2.0 + 5e-3)  # Difference of 5e-3, within 1e-2

        ref_kernel._default_inputs = [torch.randn(100, device=device)]
        custom_kernel._default_inputs = ref_kernel._default_inputs

        result = validator.validate_correctness(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            device=device,
            num_correct_trials=1
        )

        # Should pass with default tolerance
        assert result.is_correct

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_custom_stricter_tolerance_fails(self):
        """Test that stricter custom tolerance causes validation to fail"""
        validator = CorrectnessValidator()
        device = torch.device("cuda:0")

        # Create kernels with small difference
        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        custom_kernel = SimpleKernel(lambda x: x * 2.0 + 5e-3)  # Difference of 5e-3

        ref_kernel._default_inputs = [torch.randn(100, device=device)]
        custom_kernel._default_inputs = ref_kernel._default_inputs

        # Use stricter tolerance - should fail
        result = validator.validate_correctness(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            device=device,
            num_correct_trials=1,
            atol=1e-5,
            rtol=1e-5
        )

        assert not result.is_correct
        assert result.error is not None
        # Check that error message includes actual tolerances used
        assert "atol=1e-05" in result.error or "atol=0.00001" in result.error
        assert "rtol=1e-05" in result.error or "rtol=0.00001" in result.error

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_custom_looser_tolerance_passes(self):
        """Test that looser custom tolerance allows larger differences"""
        validator = CorrectnessValidator()
        device = torch.device("cuda:0")

        # Create kernels with larger difference
        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        custom_kernel = SimpleKernel(lambda x: x * 2.0 + 0.5)  # Difference of 0.5

        ref_kernel._default_inputs = [torch.randn(100, device=device)]
        custom_kernel._default_inputs = ref_kernel._default_inputs

        # Use looser tolerance - should pass
        result = validator.validate_correctness(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            device=device,
            num_correct_trials=1,
            atol=1.0,
            rtol=1.0
        )

        assert result.is_correct

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tolerance_logging(self, caplog):
        """Test that custom tolerances are logged"""
        import logging
        caplog.set_level(logging.INFO)

        validator = CorrectnessValidator()
        device = torch.device("cuda:0")

        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        custom_kernel = SimpleKernel(lambda x: x * 2.0)

        ref_kernel._default_inputs = [torch.randn(100, device=device)]
        custom_kernel._default_inputs = ref_kernel._default_inputs

        # Call with custom tolerances
        validator.validate_correctness(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            device=device,
            num_correct_trials=1,
            atol=1e-5,
            rtol=1e-5
        )

        # Check that log message about custom tolerances was emitted
        assert any("custom tolerances" in record.message.lower() for record in caplog.records)
        assert any("atol=1e-05" in record.message or "atol=0.00001" in record.message for record in caplog.records)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_logging_for_default_tolerances(self, caplog):
        """Test that default tolerances don't trigger logging"""
        import logging
        caplog.set_level(logging.INFO)

        validator = CorrectnessValidator()
        device = torch.device("cuda:0")

        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        custom_kernel = SimpleKernel(lambda x: x * 2.0)

        ref_kernel._default_inputs = [torch.randn(100, device=device)]
        custom_kernel._default_inputs = ref_kernel._default_inputs

        # Call with default tolerances (explicit)
        validator.validate_correctness(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            device=device,
            num_correct_trials=1,
            atol=1e-2,
            rtol=1e-2
        )

        # Should NOT log about custom tolerances
        assert not any("custom tolerances" in record.message.lower() for record in caplog.records)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("atol,rtol,difference,should_pass", [
        (1e-2, 1e-2, 5e-3, True),    # Within default tolerance
        (1e-5, 1e-5, 5e-3, False),   # Outside strict tolerance
        (0.1, 0.1, 5e-3, True),      # Well within loose tolerance
        (1e-3, 1e-3, 5e-3, False),   # Just outside tolerance
        (0.01, 0.01, 0.005, True),   # Edge case: exactly at boundary
    ])
    def test_tolerance_edge_cases(self, atol, rtol, difference, should_pass):
        """Test various tolerance edge cases"""
        validator = CorrectnessValidator()
        device = torch.device("cuda:0")

        ref_kernel = SimpleKernel(lambda x: x * 2.0)
        custom_kernel = SimpleKernel(lambda x: x * 2.0 + difference)

        ref_kernel._default_inputs = [torch.randn(100, device=device)]
        custom_kernel._default_inputs = ref_kernel._default_inputs

        result = validator.validate_correctness(
            ref_kernel=ref_kernel,
            custom_kernel=custom_kernel,
            device=device,
            num_correct_trials=1,
            atol=atol,
            rtol=rtol
        )

        assert result.is_correct == should_pass