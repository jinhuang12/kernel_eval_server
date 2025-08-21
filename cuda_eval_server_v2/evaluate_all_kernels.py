#!/usr/bin/env python3
"""
Parallel evaluation script for KernelBench generated kernels
Evaluates all kernels in JSON file and updates with actual compile/correct status and performance metrics
Supports parallel evaluation across multiple GPUs
"""

import os
import sys
import json
import torch
import logging
import argparse
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from datetime import datetime
from pathlib import Path
import time
import signal
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
kernelbench_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, kernelbench_dir)
sys.path.insert(0, current_dir)

# Import KernelBench eval functions
from KernelBench.eval import (
    eval_kernel_against_ref, 
    KernelExecResult,
    load_original_model_and_inputs,
    time_execution_with_cuda_event,
    set_seed
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [GPU %(gpu_id)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a custom logger that includes GPU ID
class GPULoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'gpu_id': self.extra.get('gpu_id', 'MAIN')}}

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    index: int
    compile: bool
    correct: bool
    ref_runtime_ms: float
    generated_runtime_ms: float
    speedup: float
    gpu_id: int
    evaluation_timestamp: str
    evaluation_error: Optional[str] = None


class KernelEvaluator:
    """Handles evaluation of a single kernel"""
    
    def __init__(self, gpu_id: int, build_dir: str, num_trials: int = 10, 
                 correctness_trials: int = 1, measure_performance: bool = True,
                 verbose: bool = False):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.build_dir = build_dir
        self.num_trials = num_trials
        self.correctness_trials = correctness_trials
        self.measure_performance = measure_performance
        self.verbose = verbose
        
        # Create GPU-specific logger
        self.logger = GPULoggerAdapter(logger, {'gpu_id': gpu_id})
        
    def measure_reference_runtime(self, ref_code: str, seed: int = 42) -> float:
        """Measure runtime of the reference PyTorch model"""
        try:
            # Load the reference model
            context = {}
            Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_code, context)
            
            if Model is None:
                self.logger.warning("Failed to load reference model")
                return -1.0
            
            # Initialize model
            set_seed(seed)
            init_inputs = get_init_inputs()
            init_inputs = [
                x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x 
                for x in init_inputs
            ]
            
            with torch.no_grad():
                set_seed(seed)
                model = Model(*init_inputs).cuda(device=self.device)
                
                # Get inputs
                set_seed(seed)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=self.device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                
                # Measure runtime
                torch.cuda.synchronize(device=self.device)
                elapsed_times = time_execution_with_cuda_event(
                    model,
                    *inputs,
                    num_trials=self.num_trials,
                    verbose=False,
                    device=self.device
                )
                
                # Return mean runtime in ms
                if elapsed_times:
                    return float(np.mean(elapsed_times))
                    
        except Exception as e:
            self.logger.warning(f"Failed to measure reference runtime: {e}")
        
        return -1.0
    
    def evaluate_kernel(self, index: int, ref_code: str, generated_code: str) -> EvaluationResult:
        """Evaluate a single kernel and return results"""
        
        timestamp = datetime.now().isoformat()
        
        try:
            # Build directory for this specific kernel
            kernel_build_dir = os.path.join(self.build_dir, f"kernel_{index}_gpu_{self.gpu_id}")
            os.makedirs(kernel_build_dir, exist_ok=True)
            
            self.logger.info(f"Evaluating kernel {index}")
            
            # Run evaluation
            result = eval_kernel_against_ref(
                original_model_src=ref_code,
                custom_model_src=generated_code,
                seed_num=42,
                num_correct_trials=self.correctness_trials,
                num_perf_trials=self.num_trials,
                verbose=self.verbose,
                measure_performance=self.measure_performance,
                build_dir=kernel_build_dir,
                device=self.device
            )
            
            # Handle None result (retry once if lock error)
            if result is None:
                self.logger.warning(f"Kernel {index}: Got None result, retrying once...")
                time.sleep(5)  # Longer pause before retry to avoid lock conflicts
                result = eval_kernel_against_ref(
                    original_model_src=ref_code,
                    custom_model_src=generated_code,
                    seed_num=42,
                    num_correct_trials=self.correctness_trials,
                    num_perf_trials=self.num_trials,
                    verbose=self.verbose,
                    measure_performance=self.measure_performance,
                    build_dir=kernel_build_dir,
                    device=self.device
                )
            
            if result is None:
                # Still None after retry
                return EvaluationResult(
                    index=index,
                    compile=False,
                    correct=False,
                    ref_runtime_ms=-1.0,
                    generated_runtime_ms=-1.0,
                    speedup=0.0,
                    gpu_id=self.gpu_id,
                    evaluation_timestamp=timestamp,
                    evaluation_error="Evaluation returned None (possible lock error)"
                )
            
            # Extract results
            compile_status = result.compiled
            correct_status = result.correctness
            
            # Extract runtime metrics
            ref_runtime = -1.0
            generated_runtime = result.runtime if result.runtime > 0 else -1.0
            speedup = 0.0
            
            # Calculate speedup if we have performance data
            if self.measure_performance and correct_status and generated_runtime > 0:
                # Measure reference model runtime separately
                self.logger.info(f"Kernel {index}: Measuring reference model runtime...")
                ref_runtime = self.measure_reference_runtime(ref_code, seed=42)
                
                if ref_runtime > 0 and generated_runtime > 0:
                    speedup = ref_runtime / generated_runtime
                    self.logger.info(f"Kernel {index}: Ref={ref_runtime:.3f}ms, Gen={generated_runtime:.3f}ms, Speedup={speedup:.2f}x")
                else:
                    self.logger.warning(f"Kernel {index}: Could not calculate speedup (ref={ref_runtime}, gen={generated_runtime})")
            
            # Log result
            status_str = f"Compile: {compile_status}, Correct: {correct_status}"
            if speedup > 0:
                status_str += f", Speedup: {speedup:.2f}x"
            self.logger.info(f"Kernel {index}: {status_str}")
            
            return EvaluationResult(
                index=index,
                compile=compile_status,
                correct=correct_status,
                ref_runtime_ms=ref_runtime,
                generated_runtime_ms=generated_runtime,
                speedup=speedup,
                gpu_id=self.gpu_id,
                evaluation_timestamp=timestamp,
                evaluation_error=None
            )
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Kernel {index}: Evaluation failed - {error_msg}")
            
            # Try to extract compile/correct status from error
            compile_status = False
            correct_status = False
            error_str = str(e).lower()
            
            # If it's a runtime error but not compilation, compilation succeeded
            if "runtime" in error_str and "compil" not in error_str:
                compile_status = True
            
            return EvaluationResult(
                index=index,
                compile=compile_status,
                correct=correct_status,
                ref_runtime_ms=-1.0,
                generated_runtime_ms=-1.0,
                speedup=0.0,
                gpu_id=self.gpu_id,
                evaluation_timestamp=timestamp,
                evaluation_error=error_msg
            )


def gpu_worker(gpu_id: int, work_queue: Queue, result_queue: Queue, 
               build_dir: str, args: argparse.Namespace, stop_event):
    """Worker process for a single GPU"""
    
    # Set GPU for this process
    torch.cuda.set_device(gpu_id)
    
    # Create evaluator for this GPU
    evaluator = KernelEvaluator(
        gpu_id=gpu_id,
        build_dir=build_dir,
        num_trials=args.num_trials,
        correctness_trials=args.correctness_trials,
        measure_performance=args.measure_performance,
        verbose=args.verbose
    )
    
    gpu_logger = GPULoggerAdapter(logger, {'gpu_id': gpu_id})
    gpu_logger.info(f"GPU worker started")
    
    processed_count = 0
    
    while not stop_event.is_set():
        try:
            # Get work item with timeout (increased for long compilation times)
            item = work_queue.get(timeout=10)
            
            if item is None:  # Poison pill
                break
                
            index, ref_code, generated_code = item
            
            # Evaluate kernel
            result = evaluator.evaluate_kernel(index, ref_code, generated_code)
            
            # Put result in queue
            result_queue.put(result)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                gpu_logger.info(f"Processed {processed_count} kernels")
                
        except mp.queues.Empty:
            continue
        except Exception as e:
            gpu_logger.error(f"Worker error: {e}")
            traceback.print_exc()
    
    gpu_logger.info(f"GPU worker finished - processed {processed_count} kernels")


def load_checkpoint(checkpoint_file: str) -> Dict[int, Any]:
    """Load checkpoint file if it exists"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                return {item['index']: item for item in checkpoint.get('processed', [])}
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    return {}


def save_checkpoint(checkpoint_file: str, processed_results: List[Dict]):
    """Save checkpoint file"""
    try:
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'processed_count': len(processed_results),
            'processed': processed_results
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate KernelBench kernels in parallel across GPUs')
    parser.add_argument('--input-json', type=str, 
                       default='test_data/kernelbench_generated_kernels.json',
                       help='Input JSON file path')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Output JSON file path (default: kernelbench_evaluated_{timestamp}.json)')
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated GPU IDs to use (default: auto-detect all)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                       help='Save checkpoint every N kernels')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint files')
    parser.add_argument('--num-trials', type=int, default=10,
                       help='Number of performance trials')
    parser.add_argument('--correctness-trials', type=int, default=1,
                       help='Number of correctness trials')
    parser.add_argument('--measure-performance', action='store_true', default=True,
                       help='Measure performance metrics')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start from specific index')
    parser.add_argument('--end-index', type=int, default=None,
                       help='End at specific index')
    parser.add_argument('--build-dir', type=str, default='build_cache',
                       help='Directory for build cache')
    parser.add_argument('--timeout', type=int, default=120,
                       help='Timeout for result queue (default: 120s, accounts for 50-60s compilation)')
    
    args = parser.parse_args()
    
    # Setup output file
    if args.output_json is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_json = f'kernelbench_evaluated_{timestamp}.json'
    
    # Detect GPUs
    if not torch.cuda.is_available():
        logger.error("No CUDA devices available!")
        sys.exit(1)
    
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Result timeout: {args.timeout}s (compilation typically takes 50-60s)")
    
    # Load input JSON
    logger.info(f"Loading input JSON: {args.input_json}")
    try:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input JSON: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(data)} kernels from JSON")
    
    # Apply index range if specified
    if args.end_index is not None:
        data = data[args.start_index:args.end_index]
    else:
        data = data[args.start_index:]
    
    logger.info(f"Processing {len(data)} kernels (indices {args.start_index} to {args.start_index + len(data) - 1})")
    
    # Load checkpoint if resuming
    checkpoint_file = f'checkpoint_eval.json'
    processed_checkpoint = {}
    if args.resume:
        processed_checkpoint = load_checkpoint(checkpoint_file)
        logger.info(f"Loaded {len(processed_checkpoint)} processed results from checkpoint")
    
    # Create build directory
    os.makedirs(args.build_dir, exist_ok=True)
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    manager = Manager()
    work_queue = manager.Queue()
    result_queue = manager.Queue()
    stop_event = manager.Event()
    
    # Create worker processes
    workers = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=gpu_worker, args=(
            gpu_id, work_queue, result_queue, args.build_dir, args, stop_event
        ))
        p.start()
        workers.append(p)
    
    logger.info(f"Started {len(workers)} GPU workers")
    
    # Add work items to queue (skip already processed if resuming)
    total_items = 0
    skipped_items = 0
    for i, item in enumerate(data):
        actual_index = args.start_index + i
        
        # Skip if already processed
        if args.resume and actual_index in processed_checkpoint:
            skipped_items += 1
            continue
        
        if 'ref' in item and 'generated' in item:
            work_queue.put((actual_index, item['ref'], item['generated']))
            total_items += 1
    
    logger.info(f"Added {total_items} kernels to work queue ({skipped_items} skipped from checkpoint)")
    
    # Add poison pills to stop workers when done
    for _ in workers:
        work_queue.put(None)
    
    # Collect results
    results_dict = dict(processed_checkpoint)  # Start with checkpoint data
    last_checkpoint_save = time.time()
    processed_count = len(processed_checkpoint)
    
    try:
        while processed_count < total_items + skipped_items:
            try:
                # Timeout accounts for 50-60s compilation time plus evaluation
                result = result_queue.get(timeout=args.timeout)
                
                # Convert result to dict
                result_dict = asdict(result)
                results_dict[result.index] = result_dict
                processed_count += 1
                
                # Progress logging
                if processed_count % 10 == 0:
                    progress = (processed_count / (total_items + skipped_items)) * 100
                    logger.info(f"Overall progress: {processed_count}/{total_items + skipped_items} ({progress:.1f}%)")
                
                # Save checkpoint periodically (increased interval due to longer compilation)
                if time.time() - last_checkpoint_save > 300 or processed_count % args.checkpoint_interval == 0:
                    save_checkpoint(checkpoint_file, list(results_dict.values()))
                    last_checkpoint_save = time.time()
                    logger.info(f"Checkpoint saved ({processed_count} results)")
                    
            except mp.queues.Empty:
                # Check if workers are still alive
                alive_workers = sum(1 for w in workers if w.is_alive())
                if alive_workers == 0:
                    logger.warning("All workers have stopped")
                    break
                else:
                    # Workers still running, probably compiling (can take 50-60s per kernel)
                    logger.info(f"Waiting for results... {alive_workers} workers still active (compilation can take 50-60s)")
                    continue
                    
    except KeyboardInterrupt:
        logger.info("Interrupted by user - saving checkpoint...")
        save_checkpoint(checkpoint_file, list(results_dict.values()))
        stop_event.set()
        
    finally:
        # Signal workers to stop
        stop_event.set()
        
        # Wait for workers to finish
        logger.info("Waiting for workers to finish...")
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
    
    logger.info(f"Collected {len(results_dict)} total results")
    
    # Update original data with results
    for i, item in enumerate(data):
        actual_index = args.start_index + i
        if actual_index in results_dict:
            result = results_dict[actual_index]
            # Overwrite existing fields
            item['compile'] = result['compile']
            item['correct'] = result['correct']
            # Add new fields
            item['ref_runtime_ms'] = result['ref_runtime_ms']
            item['generated_runtime_ms'] = result['generated_runtime_ms']
            item['speedup'] = result['speedup']
            item['gpu_id'] = result['gpu_id']
            item['evaluation_timestamp'] = result['evaluation_timestamp']
            if result['evaluation_error']:
                item['evaluation_error'] = result['evaluation_error']
    
    # Save final output
    logger.info(f"Saving results to {args.output_json}")
    try:
        with open(args.output_json, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Results saved successfully")
        
        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed")
            
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        sys.exit(1)
    
    # Print summary statistics
    compile_count = sum(1 for r in results_dict.values() if r['compile'])
    correct_count = sum(1 for r in results_dict.values() if r['correct'])
    speedup_values = [r['speedup'] for r in results_dict.values() if r['speedup'] > 0]
    
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total kernels evaluated: {len(results_dict)}")
    logger.info(f"Compilation successful: {compile_count} ({compile_count/len(results_dict)*100:.1f}%)")
    logger.info(f"Correctness passed: {correct_count} ({correct_count/len(results_dict)*100:.1f}%)")
    
    if speedup_values:
        logger.info(f"Average speedup: {np.mean(speedup_values):.2f}x")
        logger.info(f"Median speedup: {np.median(speedup_values):.2f}x")
        logger.info(f"Max speedup: {np.max(speedup_values):.2f}x")
    
    logger.info("=" * 60)
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
