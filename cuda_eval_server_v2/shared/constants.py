"""
Shared constants for CUDA Evaluation Server V2
Central location for default values to ensure consistency across all APIs
"""

# Request parameter defaults
DEFAULT_NUM_TRIALS = 100
DEFAULT_NUM_WARMUP = 10
DEFAULT_TIMEOUT = 120  # seconds

# Validation tolerance defaults
# CRITICAL: These must be consistent across FastAPI and MCP
DEFAULT_ATOL = 1e-2  # Absolute tolerance for numerical comparison
DEFAULT_RTOL = 1e-2  # Relative tolerance for numerical comparison

# Parameter validation bounds
MIN_NUM_TRIALS = 1
MAX_NUM_TRIALS = 10000
MIN_TIMEOUT = 5  # seconds
MAX_TIMEOUT = 600  # seconds (10 minutes)

# Profiling defaults
DEFAULT_PERCENTILES = [95, 99]
DEFAULT_ENABLE_DEVICE_METRICS = True

# GPU resource defaults
DEFAULT_GPU_ID = 0
GPU_ALLOCATION_TIMEOUT = 30  # seconds

# Subprocess defaults
SUBPROCESS_TIMEOUT_BUFFER = 10  # Extra seconds for subprocess overhead

# Serialization defaults
FLOAT_DECIMAL_PLACES = 3  # Round floats to this many decimal places in responses

# Cache defaults
DEFAULT_CUPY_CACHE_DIR = "/tmp/cupy_kernel_cache"