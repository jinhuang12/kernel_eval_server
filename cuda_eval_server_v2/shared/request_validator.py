"""
Shared request validation for CUDA Evaluation Server V2
Provides parameter validation functions used by both FastAPI and MCP
"""

from typing import Optional
from shared.constants import (
    MIN_NUM_TRIALS, MAX_NUM_TRIALS,
    MIN_TIMEOUT, MAX_TIMEOUT
)


class ValidationError(ValueError):
    """Custom exception for validation errors"""
    pass


def validate_num_trials(num_trials: int) -> None:
    """
    Validate the number of trials parameter.

    Args:
        num_trials: Number of trials to validate

    Raises:
        ValidationError: If num_trials is out of valid range
    """
    if num_trials < MIN_NUM_TRIALS:
        raise ValidationError(f"num_trials must be at least {MIN_NUM_TRIALS}")
    if num_trials > MAX_NUM_TRIALS:
        raise ValidationError(f"num_trials cannot exceed {MAX_NUM_TRIALS}")


def validate_timeout(timeout: int) -> None:
    """
    Validate the timeout parameter.

    Args:
        timeout: Timeout in seconds to validate

    Raises:
        ValidationError: If timeout is out of valid range
    """
    if timeout < MIN_TIMEOUT:
        raise ValidationError(f"timeout must be at least {MIN_TIMEOUT} seconds")
    if timeout > MAX_TIMEOUT:
        raise ValidationError(f"timeout cannot exceed {MAX_TIMEOUT} seconds")


def validate_tolerance(atol: Optional[float], rtol: Optional[float]) -> None:
    """
    Validate tolerance parameters for numerical comparison.

    Args:
        atol: Absolute tolerance (optional)
        rtol: Relative tolerance (optional)

    Raises:
        ValidationError: If tolerances are invalid
    """
    if atol is not None and atol < 0:
        raise ValidationError("atol must be non-negative")
    if rtol is not None and rtol < 0:
        raise ValidationError("rtol must be non-negative")
    if atol is not None and atol > 1.0:
        raise ValidationError("atol should typically be less than 1.0")
    if rtol is not None and rtol > 1.0:
        raise ValidationError("rtol should typically be less than 1.0")


def validate_compare_request(
    num_trials: int,
    timeout: int,
    atol: Optional[float] = None,
    rtol: Optional[float] = None
) -> None:
    """
    Validate all parameters for a kernel comparison request.

    Args:
        num_trials: Number of trials
        timeout: Timeout in seconds
        atol: Absolute tolerance (optional)
        rtol: Relative tolerance (optional)

    Raises:
        ValidationError: If any parameter is invalid
    """
    validate_num_trials(num_trials)
    validate_timeout(timeout)
    validate_tolerance(atol, rtol)


def validate_evaluation_request(num_trials: int, timeout: int) -> None:
    """
    Validate all parameters for a single kernel evaluation request.

    Args:
        num_trials: Number of trials
        timeout: Timeout in seconds

    Raises:
        ValidationError: If any parameter is invalid
    """
    validate_num_trials(num_trials)
    validate_timeout(timeout)