"""
Shared utilities for CUDA Evaluation Server V2
"""

import socket
import logging
from typing import Dict, Any, Optional, Union
from .models import TensorData, TensorSpec
import base64, zlib, numpy as np, torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server constants
HOSTNAME = socket.gethostname()

# Robust IP address resolution with fallback
try:
    IP_ADDRESS = socket.gethostbyname(HOSTNAME)
except (socket.gaierror, OSError):
    # Fallback to localhost if hostname resolution fails
    try:
        IP_ADDRESS = socket.gethostbyname('localhost')
    except (socket.gaierror, OSError):
        IP_ADDRESS = '127.0.0.1'

def create_error_response(
    error_message: str,
    job_id: str = None,
    gpu_id: int = None,
    compilation_method: str = "cupy"
) -> Dict[str, Any]:
    """Create a standardized error response that maintains API compatibility"""
    error_data = {
        "kernel_exec_result": {},
        "ref_runtime": {},
        "pod_name": HOSTNAME,
        "pod_ip": IP_ADDRESS,
        "status": "error",
        "error": error_message,
        "compilation_method": compilation_method,
    }
    
    if job_id is not None:
        error_data["job_id"] = job_id
        
    if gpu_id is not None:
        error_data["gpu_id"] = gpu_id

    return error_data

def create_no_cache_headers() -> Dict[str, str]:
    """Create headers to prevent caching - maintains compatibility with existing API"""
    return {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }


# --- dtype maps ---
_TORCH2NP = {
    torch.float64: np.float64, torch.float32: np.float32, torch.float16: np.float16,
    torch.bfloat16: np.float16,  # store bfloat16 via np.float16 if needed, or use np.uint16 raw
    torch.int64: np.int64, torch.int32: np.int32, torch.int16: np.int16,
    torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: np.bool_,
}
_STR2TORCH = {
    "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "int64": torch.int64, "int32": torch.int32,
    "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool,
}
_STR2NP = {k: v for v, k in [(np.float64, "float64"), (np.float32, "float32"),
                              (np.float16, "float16"), (np.int64, "int64"), (np.int32, "int32"),
                              (np.int16, "int16"), (np.int8, "int8"), (np.uint8, "uint8"), (np.bool_, "bool")]}

def encode_tensor_to_data(t: torch.Tensor, compress: str = "none") -> TensorData:
    """
    Convert a torch tensor to JSON-friendly TensorData (base64).
    - Always encodes CPU, contiguous storage.
    - Preserves dtype and shape; ignores strides (contiguous in payload).
    """
    if t.is_cuda:
        t = t.detach().cpu()
    arr = t.detach().contiguous().numpy()
    raw = arr.tobytes(order="C")
    if compress == "zlib":
        raw = zlib.compress(raw)
    b64 = base64.b64encode(raw).decode("ascii")
    return TensorData(
        data_b64=b64, compress=compress, shape=list(arr.shape),
        dtype=str(arr.dtype)
    )

def decode_tensor_from_data(td: TensorData, device: str | torch.device = "cuda") -> torch.Tensor:
    """
    Decode JSON TensorData back to a torch tensor on `device`.
    - Assumes row-major when td.strides is None.
    - If td.strides provided, constructs a view using as_strided (requires enough storage).
    """
    buf = base64.b64decode(td.data_b64)
    if td.compress == "zlib":
        buf = zlib.decompress(buf)
    np_dt = _STR2NP[td.dtype]
    arr = np.frombuffer(buf, dtype=np_dt)
    # If no custom storage layout, reshape C-contiguously
    arr = arr.reshape(td.shape)
    t = torch.from_numpy(arr.copy())  # own memory
    if isinstance(device, str):
        device = torch.device(device)
    return t.to(device=device)

def materialize_tensor(spec: TensorSpec, default_device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Create a tensor from spec (prefer literal data; else generate via init)."""
    if spec.data is not None:
        # Use device from spec if available, otherwise use default
        device = getattr(spec, 'device', default_device or 'cuda')
        return decode_tensor_from_data(spec.data, device=device)
    # generate by init (deterministic if seed set)
    dt = _STR2TORCH[spec.dtype]
    # Use device from spec if available, otherwise use default
    device_str = getattr(spec, 'device', default_device or 'cuda')
    if isinstance(device_str, torch.device):
        dev = device_str
    else:
        dev = torch.device(device_str)
    shape = tuple(int(s) for s in spec.shape)
    if spec.init is None or spec.init.kind == "randn":
        g = torch.Generator(device=dev)
        if spec.init and spec.init.seed is not None: g.manual_seed(int(spec.init.seed))
        t = torch.randn(*shape, dtype=dt, device=dev, generator=g)
        if spec.init and spec.init.mean is not None:
            mean = spec.init.mean or 0.0; std = spec.init.std or 1.0
            t = t * std + mean
        return t
    kind = spec.init.kind
    if kind == "zeros": return torch.zeros(shape, dtype=dt, device=dev)
    if kind == "ones":  return torch.ones(shape, dtype=dt, device=dev)
    if kind == "full":
        val = float(spec.init.fill_value or 0.0); return torch.full(shape, val, dtype=dt, device=dev)
    if kind == "uniform":
        low = float(spec.init.low or 0.0); high = float(spec.init.high or 1.0)
        g = torch.Generator(device=dev)
        if spec.init.seed is not None: g.manual_seed(int(spec.init.seed))
        return (low + (high - low) * torch.rand(*shape, dtype=dt, device=dev, generator=g))
    if kind == "arange":
        start = float(spec.init.start or 0.0); step = float(spec.init.step or 1.0)
        # arange returns 1D; tile to shape if needed
        v = torch.arange(start, start + step * shape[-1], step=step, dtype=dt, device=dev)[:shape[-1]]
        return v.reshape([1] * (len(shape) - 1) + [shape[-1]]).expand(shape).clone()
    raise ValueError(f"Unsupported init kind: {spec.init.kind}")
