"""
Shared utilities for CUDA Evaluation Server V2
"""

import socket
import logging
from typing import Dict, Any, Optional, Union

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