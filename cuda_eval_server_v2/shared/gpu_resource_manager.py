"""
Centralized GPU resource management for compilation and profiling services
Single instance created at JobManager level and injected into services
"""

import asyncio
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager
from queue import Queue, Empty
import torch

from shared.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """Centralized GPU allocation manager"""
    
    def __init__(self):
        self._gpu_queue = Queue()
        self.metrics_collector = get_metrics_collector()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self._gpu_queue.put(i)
        logger.info(f"GPUResourceManager initialized with {torch.cuda.device_count()} GPUs")
    
    @asynccontextmanager
    async def acquire_gpu(self, timeout: int = 300, job_id: Optional[str] = None):
        """Acquire GPU resource with proper error handling"""
        loop = asyncio.get_event_loop()
        acquire_start = time.time()
        gpu_id = None
        
        try:
            gpu_id = await loop.run_in_executor(None, lambda: self._gpu_queue.get(timeout=timeout))
            wait_time = time.time() - acquire_start
            
            if job_id:
                self.metrics_collector.record_gpu_acquisition(job_id, gpu_id, wait_time)
            
            logger.info(f"Acquired GPU {gpu_id} after {wait_time:.2f}s wait")
            yield gpu_id
            
        except Empty:
            logger.error(f"GPU acquisition timeout after {timeout}s for job {job_id}")
            raise TimeoutError(f"No GPU available within {timeout} seconds")
        except Exception as e:
            logger.error(f"GPU acquisition error for job {job_id}: {e}")
            raise
        finally:
            if gpu_id is not None:
                try:
                    if job_id:
                        self.metrics_collector.record_gpu_release(job_id)
                    self._gpu_queue.put(gpu_id)
                    logger.info(f"Released GPU {gpu_id}")
                except Exception as e:
                    logger.error(f"Error releasing GPU {gpu_id}: {e}")
                    # Recovery: re-add GPU to prevent permanent loss
                    try:
                        self._gpu_queue.put(gpu_id)
                    except:
                        pass
    
    def get_gpu_count(self) -> int:
        """Get total number of available GPUs"""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    def get_available_gpu_count(self) -> int:
        """Get number of currently available GPUs"""
        return self._gpu_queue.qsize()
