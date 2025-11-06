#!/usr/bin/env python3
"""
Comprehensive Test Suite for CUDA Evaluation Server V2
Modular, maintainable, and efficient testing framework
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from test_utils import (
    TestConfig, TestMode, TestResult, TestCaseProvider,
    ServerClient, ResultAnalyzer, MetricsCollector, TestReporter,
    create_evaluation_request, logger, KernelType, TritonTestCase,
    IOContractTestProvider
)


def format_device_metrics(metrics: Dict[str, any]):
    """Format and print device metrics"""
    if not metrics:
        print("  No device metrics available")
        return
    
    # Check if we have any actual metrics data
    has_original = "original_device_metrics" in metrics and metrics["original_device_metrics"]
    has_custom = "custom_device_metrics" in metrics and metrics["custom_device_metrics"]
    
    if not has_original and not has_custom:
        print("  Device metrics collected but no data available")
        return
    
    # Original metrics
    if has_original:
        orig = metrics["original_device_metrics"]
        print("  📊 ORIGINAL (Reference) METRICS:")
        
        if "speed_of_light" in orig and orig["speed_of_light"]:
            sol = orig["speed_of_light"]
            print(f"    Speed of Light Analysis:")
            print(f"      • Compute Throughput: {sol.get('compute_throughput_pct', 'N/A')}%")
            print(f"      • Memory Throughput: {sol.get('memory_throughput_pct', 'N/A')}%")
        
        if "detailed_metrics" in orig:
            detailed = orig["detailed_metrics"]
            print(f"    Detailed Metrics:")
            if "l1_hit_rate_pct" in detailed:
                print(f"      • L1 Cache Hit Rate: {detailed['l1_hit_rate_pct']:.1f}%")
            if "l2_hit_rate_pct" in detailed:
                print(f"      • L2 Cache Hit Rate: {detailed['l2_hit_rate_pct']:.1f}%")
            if "warp_occupancy_pct" in detailed:
                print(f"      • Warp Occupancy: {detailed['warp_occupancy_pct']:.1f}%")
            if "instructions_per_cycle" in detailed:
                print(f"      • Instructions Per Cycle: {detailed['instructions_per_cycle']:.2f}")
    
    # Custom metrics
    if has_custom:
        custom = metrics["custom_device_metrics"]
        print("\n  📊 CUSTOM (Generated) METRICS:")
        
        if "speed_of_light" in custom and custom["speed_of_light"]:
            sol = custom["speed_of_light"]
            print(f"    Speed of Light Analysis:")
            print(f"      • Compute Throughput: {sol.get('compute_throughput_pct', 'N/A')}%")
            print(f"      • Memory Throughput: {sol.get('memory_throughput_pct', 'N/A')}%")
        
        if "detailed_metrics" in custom:
            detailed = custom["detailed_metrics"]
            print(f"    Detailed Metrics:")
            if "l1_hit_rate_pct" in detailed:
                print(f"      • L1 Cache Hit Rate: {detailed['l1_hit_rate_pct']:.1f}%")
            if "l2_hit_rate_pct" in detailed:
                print(f"      • L2 Cache Hit Rate: {detailed['l2_hit_rate_pct']:.1f}%")
            if "warp_occupancy_pct" in detailed:
                print(f"      • Warp Occupancy: {detailed['warp_occupancy_pct']:.1f}%")
            if "instructions_per_cycle" in detailed:
                print(f"      • Instructions Per Cycle: {detailed['instructions_per_cycle']:.2f}")
    
    # Performance comparison
    if has_original and has_custom:
        orig_sol = metrics.get("original_device_metrics", {}).get("speed_of_light", {})
        custom_sol = metrics.get("custom_device_metrics", {}).get("speed_of_light", {})
        
        if orig_sol and custom_sol:
            orig_compute = orig_sol.get("compute_throughput_pct")
            custom_compute = custom_sol.get("compute_throughput_pct")
            orig_memory = orig_sol.get("memory_throughput_pct")
            custom_memory = custom_sol.get("memory_throughput_pct")
            
            if all(v is not None for v in [orig_compute, custom_compute, orig_memory, custom_memory]):
                print("\n  📊 PERFORMANCE COMPARISON:")
                
                # Compute comparison
                compute_diff = custom_compute - orig_compute
                compute_sign = "+" if compute_diff > 0 else ""
                print(f"    • Compute Throughput: {compute_sign}{compute_diff:.1f}% "
                      f"({orig_compute:.1f}% → {custom_compute:.1f}%)")
                
                # Memory comparison
                memory_diff = custom_memory - orig_memory
                memory_sign = "+" if memory_diff > 0 else ""
                print(f"    • Memory Throughput: {memory_sign}{memory_diff:.1f}% "
                      f"({orig_memory:.1f}% → {custom_memory:.1f}%)")
                
                # Bottleneck analysis
                print("\n  🔍 BOTTLENECK ANALYSIS:")
                if orig_compute > orig_memory:
                    print(f"    • Original: Compute-bound ({orig_compute:.1f}% > {orig_memory:.1f}%)")
                else:
                    print(f"    • Original: Memory-bound ({orig_memory:.1f}% ≥ {orig_compute:.1f}%)")
                
                if custom_compute > custom_memory:
                    print(f"    • Custom: Compute-bound ({custom_compute:.1f}% > {custom_memory:.1f}%)")
                else:
                    print(f"    • Custom: Memory-bound ({custom_memory:.1f}% ≥ {custom_compute:.1f}%)")


class FunctionalTests:
    """Functional testing module"""
    
    def __init__(self, client: ServerClient, provider: TestCaseProvider, config: TestConfig):
        self.client = client
        self.provider = provider
        self.config = config
    
    def test_basic_functionality(self) -> Dict[str, any]:
        """Test basic server functionality"""
        logger.info("Testing basic server functionality...")
        metrics = MetricsCollector()
        metrics.start()
        
        # Health check
        success, health_data = self.client.health_check()
        metrics.add_result(TestResult(
            test_name="health_check",
            success=success,
            duration=0.1,
            metadata=health_data
        ))
        
        # Stats endpoint
        stats_data = self.client.get_stats()
        metrics.add_result(TestResult(
            test_name="stats_endpoint",
            success="error" not in stats_data,
            duration=0.1,
            metadata=stats_data
        ))
        
        # Simple evaluation
        test_case = self.provider.get_random_case()
        if test_case:
            request = create_evaluation_request(test_case, num_trials=1)
            start = time.time()
            success, response = self.client.submit_evaluation(request)
            duration = time.time() - start
            
            metrics.add_result(TestResult(
                test_name="simple_evaluation",
                success=success,
                duration=duration,
                status_code=response.get("status_code", 200),
                error=response.get("error")
            ))
        
        metrics.stop()
        return metrics.get_summary()
    
    def test_validation_accuracy(self, batch_size: int = 20) -> Dict[str, any]:
        """Test validation accuracy against expected results"""
        logger.info(f"Testing validation accuracy with {batch_size} cases...")
        metrics = MetricsCollector()
        metrics.start()
        
        test_batch = self.provider.get_batch(batch_size)
        validation_correct = 0
        
        for test_case in test_batch:
            request = create_evaluation_request(test_case, self.config.num_trials)
            if self.config.enable_device_metrics:
                request["enable_device_metrics"] = True
            start = time.time()
            success, response = self.client.submit_evaluation(request)
            duration = time.time() - start
            
            if success:
                actual_compile = ResultAnalyzer.extract_compilation_status(response)
                actual_correct = ResultAnalyzer.extract_correctness_status(response)
                
                # Check if actuals match expectations or are better
                expected_compile = test_case.get("expected_compile")
                expected_correct = test_case.get("expected_correct")
                
                # Exact match
                compile_match = (expected_compile == actual_compile 
                               if expected_compile is not None else True)
                correct_match = (expected_correct == actual_correct
                               if expected_correct is not None else True)
                
                # Better than expected (expected false but got true)
                compile_better = (expected_compile is False and actual_compile is True)
                correct_better = (expected_correct is False and actual_correct is True)
                
                # Pass if exact match OR better than expected
                validation_passed = (compile_match and correct_match) or compile_better or correct_better
                if validation_passed:
                    validation_correct += 1
                
                result = TestResult(
                    test_name=f"validation_{test_case.get('index', 'unknown')}",
                    success=success,
                    duration=duration,
                    expected_compile=test_case.get("expected_compile"),
                    expected_correct=test_case.get("expected_correct"),
                    actual_compile=actual_compile,
                    actual_correct=actual_correct
                )
            else:
                result = TestResult(
                    test_name=f"validation_{test_case.get('index', 'unknown')}",
                    success=False,
                    duration=duration,
                    error=response.get("error")
                )
            
            metrics.add_result(result)
        
        metrics.stop()
        summary = metrics.get_summary()
        summary["validation_accuracy"] = validation_correct / len(test_batch) if test_batch else 0
        return summary
    
    def test_error_handling(self) -> Dict[str, any]:
        """Test error handling with malformed requests"""
        logger.info("Testing error handling...")
        metrics = MetricsCollector()
        metrics.start()
        
        # Test cases for error handling
        error_cases = [
            {
                "name": "empty_request",
                "request": {}
            },
            {
                "name": "missing_custom_code",
                "request": {"ref_code": "import torch", "num_trials": 1}
            },
            {
                "name": "invalid_kernel_type",
                "request": {
                    "ref_code": "import torch",
                    "custom_code": "import torch",
                    "kernel_type": "invalid_type"
                }
            },
            {
                "name": "syntax_error",
                "request": {
                    "ref_code": "import torch\nclass Model",  # Syntax error
                    "custom_code": "import torch",
                    "num_trials": 1
                }
            },
            {
                "name": "timeout_test",
                "request": {
                    "ref_code": "import torch\nimport time\ntime.sleep(1000)",
                    "custom_code": "import torch",
                    "num_trials": 1,
                    "timeout": 1
                }
            }
        ]
        
        for case in error_cases:
            start = time.time()
            success, response = self.client.submit_evaluation(case["request"])
            duration = time.time() - start
            
            # For error handling tests, we expect them to fail gracefully
            handled_gracefully = not success or response.get("status") == "error"
            
            metrics.add_result(TestResult(
                test_name=case["name"],
                success=handled_gracefully,
                duration=duration,
                error=response.get("error") if not handled_gracefully else None
            ))
        
        metrics.stop()
        return metrics.get_summary()


class PerformanceTests:
    """Performance testing module"""
    
    def __init__(self, client: ServerClient, provider: TestCaseProvider, config: TestConfig):
        self.client = client
        self.provider = provider
        self.config = config
    
    def test_concurrent_load(self, num_concurrent: int = None) -> Dict[str, any]:
        """Test concurrent request handling"""
        if num_concurrent is None:
            num_concurrent = self.config.concurrent_requests
        
        logger.info(f"Testing concurrent load with {num_concurrent} requests...")
        metrics = MetricsCollector()
        metrics.start()
        
        def submit_request(index: int) -> TestResult:
            test_case = self.provider.get_random_case(cuda_only=True)
            if not test_case:
                return TestResult(
                    test_name=f"concurrent_{index}",
                    success=False,
                    duration=0,
                    error="No test case available"
                )
            
            request = create_evaluation_request(test_case, self.config.num_trials)
            if self.config.enable_device_metrics:
                request["enable_device_metrics"] = True
            start = time.time()
            success, response = self.client.submit_evaluation(request)
            duration = time.time() - start
            
            perf_metrics = ResultAnalyzer.extract_performance_metrics(response) if success else {}
            
            return TestResult(
                test_name=f"concurrent_{index}",
                success=success,
                duration=duration,
                job_id=response.get("job_id"),
                gpu_id=response.get("kernel_exec_result", {}).get("metadata", {}).get("gpu_id"),
                **perf_metrics
            )
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(submit_request, i) for i in range(num_concurrent)]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.config.timeout)
                    metrics.add_result(result)
                except Exception as e:
                    metrics.add_result(TestResult(
                        test_name="concurrent_error",
                        success=False,
                        duration=0,
                        error=str(e)
                    ))
        
        metrics.stop()
        return metrics.get_summary()
    
    def test_gpu_resource_management(self) -> Dict[str, any]:
        """Test GPU resource acquisition and release"""
        logger.info("Testing GPU resource management...")
        
        # Get initial state
        initial_stats = self.client.get_stats()
        
        # Run some requests
        load_results = self.test_concurrent_load(num_concurrent=5)
        
        # Wait for cleanup
        time.sleep(2)
        
        # Get final state
        final_stats = self.client.get_stats()
        
        # Check for resource leaks
        initial_gpu = initial_stats.get("gpu_utilization", {}).get("active_sessions", 0)
        final_gpu = final_stats.get("gpu_utilization", {}).get("active_sessions", 0)
        
        return {
            "initial_active_sessions": initial_gpu,
            "final_active_sessions": final_gpu,
            "resource_leak_detected": final_gpu > initial_gpu,
            "load_test_results": load_results
        }
    
    def test_compilation_cache(self) -> Dict[str, any]:
        """Test compilation cache effectiveness"""
        logger.info("Testing compilation cache...")
        metrics = MetricsCollector()
        metrics.start()
        
        # Use the same test case multiple times
        test_case = self.provider.get_random_case(cuda_only=True)
        if not test_case:
            return {"error": "No test case available"}
        
        request = create_evaluation_request(test_case, num_trials=1)
        
        # First request (cache miss)
        start = time.time()
        success1, response1 = self.client.submit_evaluation(request)
        duration1 = time.time() - start
        
        # Second request (should be cache hit)
        start = time.time()
        success2, response2 = self.client.submit_evaluation(request)
        duration2 = time.time() - start
        
        # Third request (definitely cache hit)
        start = time.time()
        success3, response3 = self.client.submit_evaluation(request)
        duration3 = time.time() - start
        
        cache_speedup = duration1 / duration2 if duration2 > 0 else 0
        
        metrics.add_result(TestResult(
            test_name="cache_miss",
            success=success1,
            duration=duration1
        ))
        metrics.add_result(TestResult(
            test_name="cache_hit_1",
            success=success2,
            duration=duration2
        ))
        metrics.add_result(TestResult(
            test_name="cache_hit_2",
            success=success3,
            duration=duration3
        ))
        
        metrics.stop()
        summary = metrics.get_summary()
        summary["cache_effectiveness"] = {
            "cache_speedup": cache_speedup,
            "first_request": duration1,
            "cached_request": duration2,
            "improvement": f"{(1 - duration2/duration1)*100:.1f}%" if duration1 > 0 else "N/A"
        }
        return summary
    
    def test_sustained_load(self, duration_minutes: float = 2, requests_per_minute: int = 6) -> Dict[str, any]:
        """Test sustained load over time"""
        logger.info(f"Testing sustained load for {duration_minutes} minutes...")
        metrics = MetricsCollector()
        metrics.start()
        
        end_time = time.time() + (duration_minutes * 60)
        interval = 60.0 / requests_per_minute
        request_count = 0
        
        while time.time() < end_time:
            request_start = time.time()
            
            test_case = self.provider.get_random_case(cuda_only=True)
            if test_case:
                request = create_evaluation_request(test_case, num_trials=5)
                success, response = self.client.submit_evaluation(request)
                
                perf_metrics = ResultAnalyzer.extract_performance_metrics(response) if success else {}
                
                metrics.add_result(TestResult(
                    test_name=f"sustained_{request_count}",
                    success=success,
                    duration=time.time() - request_start,
                    **perf_metrics
                ))
            
            request_count += 1
            
            # Wait for next interval
            elapsed = time.time() - request_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        metrics.stop()
        summary = metrics.get_summary()
        summary["sustained_load"] = {
            "total_requests": request_count,
            "duration_minutes": duration_minutes,
            "requests_per_minute": request_count / duration_minutes
        }
        return summary


class IntegrationTests:
    """Integration testing module"""
    
    def __init__(self, client: ServerClient, provider: TestCaseProvider, config: TestConfig):
        self.client = client
        self.provider = provider
        self.config = config
    
    def test_specific_case(self, index: int, enable_device_metrics: bool = False) -> Dict[str, any]:
        """Test a specific case by index"""
        logger.info(f"Testing specific case at index {index}...")
        
        test_case = self.provider.get_case(index)
        if not test_case:
            return {"error": f"Test case {index} not found"}
        
        # Create request with device metrics flag if needed
        request = create_evaluation_request(test_case, self.config.num_trials)
        if enable_device_metrics:
            request["enable_device_metrics"] = True
        
        start = time.time()
        success, response = self.client.submit_evaluation(request)
        duration = time.time() - start
        
        if success:
            actual_compile = ResultAnalyzer.extract_compilation_status(response)
            actual_correct = ResultAnalyzer.extract_correctness_status(response)
            perf_metrics = ResultAnalyzer.extract_performance_metrics(response)
            device_metrics = ResultAnalyzer.extract_device_metrics(response)
            
            expected_compile = test_case.get("expected_compile")
            expected_correct = test_case.get("expected_correct")
            
            # Check for exact match
            compile_match = expected_compile == actual_compile if expected_compile is not None else True
            correct_match = expected_correct == actual_correct if expected_correct is not None else True
            
            # Check if better than expected
            compile_better = (expected_compile is False and actual_compile is True)
            correct_better = (expected_correct is False and actual_correct is True)
            better_than_expected = compile_better or correct_better
            
            # Pass if exact match OR better than expected
            validation_passed = (compile_match and correct_match) or better_than_expected
            
            return {
                "index": index,
                "success": True,
                "duration": duration,
                "expected": {
                    "compile": expected_compile,
                    "correct": expected_correct
                },
                "actual": {
                    "compile": actual_compile,
                    "correct": actual_correct
                },
                "performance": perf_metrics,
                "device_metrics": device_metrics,
                "validation_passed": validation_passed,
                "better_than_expected": better_than_expected
            }
        else:
            return {
                "index": index,
                "success": False,
                "duration": duration,
                "error": response.get("error")
            }
    
    def test_multi_kernel_types(self) -> Dict[str, any]:
        """Test different kernel types (TORCH, TORCH_CUDA, TRITON)"""
        logger.info("Testing multiple kernel types...")
        metrics = MetricsCollector()
        metrics.start()
        
        test_case = self.provider.get_random_case()
        if not test_case:
            return {"error": "No test case available"}
        
        kernel_types = ["torch", "torch_cuda"]  # Add "triton" if available
        
        for kernel_type in kernel_types:
            request = create_evaluation_request(
                test_case, 
                self.config.num_trials,
                kernel_type=kernel_type
            )
            
            start = time.time()
            success, response = self.client.submit_evaluation(request)
            duration = time.time() - start
            
            perf_metrics = ResultAnalyzer.extract_performance_metrics(response) if success else {}
            
            metrics.add_result(TestResult(
                test_name=f"kernel_type_{kernel_type}",
                success=success,
                duration=duration,
                metadata={"kernel_type": kernel_type},
                **perf_metrics
            ))
        
        metrics.stop()
        return metrics.get_summary()
    
    def test_stress_patterns(self) -> Dict[str, any]:
        """Test various stress patterns"""
        logger.info("Testing stress patterns...")
        
        gpu_count = self.client.get_gpu_count()
        if gpu_count == 0:
            return {"error": "No GPUs available"}
        
        results = {}
        
        # Pattern 1: Burst load (3x GPU count)
        perf_tests = PerformanceTests(self.client, self.provider, self.config)
        results["burst"] = perf_tests.test_concurrent_load(num_concurrent=gpu_count * 3)
        
        # Pattern 2: Sustained medium load
        results["sustained"] = perf_tests.test_sustained_load(
            duration_minutes=1, 
            requests_per_minute=gpu_count * 2
        )
        
        # Pattern 3: Cache stress
        results["cache"] = perf_tests.test_compilation_cache()
        
        return {
            "gpu_count": gpu_count,
            "patterns": results
        }


class MixedKernelTests:
    """Mixed kernel type testing module"""
    
    def __init__(self, client: ServerClient, config: TestConfig):
        self.client = client
        self.config = config
        self.torch_cuda_provider = TestCaseProvider(config.kernelbench_json)
        self.iocontract_provider = IOContractTestProvider(config.triton_test_dir)
    
    def test_triton_iocontract(self, batch_size: int = 5) -> Dict[str, any]:
        """Test Triton kernels with IOContract specifications"""
        logger.info(f"Testing Triton IOContract kernels (batch size: {batch_size})...")
        metrics = MetricsCollector()
        metrics.start()
        
        # Get IOContract test cases
        all_tests = self.iocontract_provider.get_all_test_cases()
        test_batch = all_tests[:batch_size] if len(all_tests) > batch_size else all_tests
        
        for test_case in test_batch:
            request = create_evaluation_request(
                test_case,
                self.config.num_trials,
                timeout=15,  # Shorter timeout for Triton tests
                enable_device_metrics=self.config.enable_device_metrics
            )
            
            start = time.time()
            success, response = self.client.submit_evaluation(request)
            duration = time.time() - start
            
            if success:
                perf_metrics = ResultAnalyzer.extract_performance_metrics(response)
                correctness = response.get('kernel_exec_result', {}).get('correctness', False)
                
                result = TestResult(
                    test_name=f"triton_io_{test_case.name}",
                    success=True,
                    duration=duration,
                    actual_correct=correctness,
                    **perf_metrics
                )
            else:
                result = TestResult(
                    test_name=f"triton_io_{test_case.name}",
                    success=False,
                    duration=duration,
                    error=response.get("error")
                )
            
            metrics.add_result(result)
        
        metrics.stop()
        summary = metrics.get_summary()
        summary["test_type"] = "triton_iocontract"
        summary["num_tests"] = len(test_batch)
        return summary
    
    def test_mixed_kernel_batch(self, total_size: int = 15) -> Dict[str, any]:
        """Test a mixed batch of different kernel types"""
        logger.info(f"Testing mixed kernel batch (size: {total_size})...")
        metrics = MetricsCollector()
        metrics.start()
        
        results_by_type = {
            KernelType.TORCH_CUDA: [],
            KernelType.TRITON: []
        }
        
        # Prepare test cases from different sources
        torch_cuda_cases = self.torch_cuda_provider.get_batch(total_size // 2, cuda_only=True)
        triton_cases = self.iocontract_provider.get_all_test_cases()[:total_size // 2]
        
        # Interleave test cases (round-robin)
        all_cases = []
        for i in range(max(len(torch_cuda_cases), len(triton_cases))):
            if i < len(torch_cuda_cases):
                all_cases.append((KernelType.TORCH_CUDA, torch_cuda_cases[i]))
            if i < len(triton_cases):
                all_cases.append((KernelType.TRITON, triton_cases[i]))
        
        # Execute mixed batch
        for kernel_type, test_case in all_cases[:total_size]:
            # Create appropriate request based on kernel type
            if kernel_type == KernelType.TORCH_CUDA:
                request = create_evaluation_request(
                    test_case,
                    self.config.num_trials,
                    timeout=self.config.timeout,  # Use config timeout for torch_cuda
                    kernel_type="torch_cuda",
                    enable_device_metrics=self.config.enable_device_metrics
                )
            else:  # TRITON
                request = create_evaluation_request(
                    test_case,
                    self.config.num_trials,
                    timeout=15,  # Shorter timeout for Triton tests
                    enable_device_metrics=self.config.enable_device_metrics
                )
            
            start = time.time()
            success, response = self.client.submit_evaluation(request)
            duration = time.time() - start
            
            # Handle both dict and TritonTestCase objects
            if hasattr(test_case, 'name'):
                case_id = test_case.name
            elif isinstance(test_case, dict):
                case_id = test_case.get('index', 'unknown')
            else:
                case_id = 'unknown'
            test_name = f"{kernel_type.value}_{case_id}"
            
            if success:
                perf_metrics = ResultAnalyzer.extract_performance_metrics(response)
                result = TestResult(
                    test_name=test_name,
                    success=True,
                    duration=duration,
                    **perf_metrics
                )
                results_by_type[kernel_type].append(result)
            else:
                result = TestResult(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=response.get("error")
                )
                results_by_type[kernel_type].append(result)
            
            metrics.add_result(result)
        
        metrics.stop()
        summary = metrics.get_summary()
        
        # Add kernel-type specific statistics
        summary["by_kernel_type"] = {}
        for kernel_type, type_results in results_by_type.items():
            if type_results:
                successful = [r for r in type_results if r.success]
                summary["by_kernel_type"][kernel_type.value] = {
                    "total": len(type_results),
                    "successful": len(successful),
                    "success_rate": len(successful) / len(type_results) if type_results else 0
                }
        
        return summary
    
    def test_kernel_comparison(self, operation: str = "add") -> Dict[str, any]:
        """Compare the same operation across different kernel types"""
        logger.info(f"Comparing '{operation}' operation across kernel types...")
        
        results = {}
        
        # Test with TORCH_CUDA if available
        if KernelType.TORCH_CUDA in self.config.kernel_types:
            # Find a matching test case
            test_case = self.torch_cuda_provider.get_random_case(cuda_only=True)
            if test_case:
                request = create_evaluation_request(
                    test_case,
                    self.config.num_trials,
                    kernel_type="torch_cuda",
                    enable_device_metrics=self.config.enable_device_metrics
                )
                success, response = self.client.submit_evaluation(request)
                if success:
                    perf = ResultAnalyzer.extract_performance_metrics(response)
                    results["torch_cuda"] = {
                        "runtime_ms": perf.get("custom_runtime_ms", 0),
                        "speedup": perf.get("speedup", 0)
                    }
        
        # Test with TRITON if available
        if KernelType.TRITON in self.config.kernel_types:
            # Find elementwise_add test case
            triton_test = self.iocontract_provider.get_test_case("elementwise_add_io")
            if triton_test:
                request = create_evaluation_request(
                    triton_test,
                    self.config.num_trials,
                    timeout=15,  # Shorter timeout for Triton tests
                    enable_device_metrics=self.config.enable_device_metrics
                )
                success, response = self.client.submit_evaluation(request)
                if success:
                    perf = ResultAnalyzer.extract_performance_metrics(response)
                    results["triton"] = {
                        "runtime_ms": perf.get("custom_runtime_ms", 0),
                        "speedup": perf.get("speedup", 0)
                    }
        
        # Compare results
        if len(results) > 1:
            runtimes = {k: v["runtime_ms"] for k, v in results.items()}
            fastest = min(runtimes, key=runtimes.get)
            results["comparison"] = {
                "fastest": fastest,
                "runtime_comparison": runtimes,
                "relative_performance": {
                    k: runtimes[fastest] / runtimes[k] if runtimes[k] > 0 else 0
                    for k in runtimes
                }
            }
        
        return results


class ComprehensiveTestSuite:
    """Main test suite orchestrator"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.client = ServerClient(config.base_url, config.timeout)
        self.provider = TestCaseProvider(config.kernelbench_json)
        
        # Test modules
        self.functional = FunctionalTests(self.client, self.provider, config)
        self.performance = PerformanceTests(self.client, self.provider, config)
        self.integration = IntegrationTests(self.client, self.provider, config)
        self.mixed_kernel = MixedKernelTests(self.client, config)
    
    def run_basic_tests(self) -> Dict[str, any]:
        """Run basic functionality tests"""
        return self.functional.test_basic_functionality()
    
    def run_validation_tests(self) -> Dict[str, any]:
        """Run validation accuracy tests"""
        results = {}
        results["validation_accuracy"] = self.functional.test_validation_accuracy(
            batch_size=self.config.batch_size
        )
        results["error_handling"] = self.functional.test_error_handling()
        return results
    
    def run_performance_tests(self) -> Dict[str, any]:
        """Run performance tests"""
        results = {}
        results["concurrent_load"] = self.performance.test_concurrent_load()
        results["gpu_management"] = self.performance.test_gpu_resource_management()
        results["cache_effectiveness"] = self.performance.test_compilation_cache()
        
        if self.config.test_mode == TestMode.COMPREHENSIVE:
            results["sustained_load"] = self.performance.test_sustained_load()
        
        return results
    
    def run_integration_tests(self) -> Dict[str, any]:
        """Run integration tests"""
        results = {}
        results["multi_kernel"] = self.integration.test_multi_kernel_types()
        
        if self.config.test_mode == TestMode.COMPREHENSIVE:
            results["stress_patterns"] = self.integration.test_stress_patterns()
        
        return results
    
    def run_triton_tests(self) -> Dict[str, any]:
        """Run Triton-specific tests"""
        results = {}
        
        # Test IOContract kernels
        if self.config.enable_iocontract:
            results["iocontract"] = self.mixed_kernel.test_triton_iocontract()
        
        # Test mixed kernel batches
        if KernelType.TRITON in self.config.kernel_types:
            results["mixed_batch"] = self.mixed_kernel.test_mixed_kernel_batch()
        
        # Kernel comparison
        if len(self.config.kernel_types) > 1:
            results["kernel_comparison"] = self.mixed_kernel.test_kernel_comparison()
        
        return results
    
    def run_comprehensive_suite(self) -> Dict[str, any]:
        """Run all tests"""
        logger.info("Running comprehensive test suite...")
        
        results = {
            "basic": self.run_basic_tests(),
            "validation": self.run_validation_tests(),
            "performance": self.run_performance_tests(),
            "integration": self.run_integration_tests()
        }
        
        # Add Triton tests if configured
        if KernelType.TRITON in self.config.kernel_types or self.config.enable_iocontract:
            results["triton"] = self.run_triton_tests()
        
        # Calculate overall statistics
        total_tests = 0
        total_passed = 0
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for test_name, test_result in category_results.items():
                    if isinstance(test_result, dict) and "total_tests" in test_result:
                        total_tests += test_result["total_tests"]
                        total_passed += test_result.get("successful", 0)
        
        results["overall"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "overall_success_rate": total_passed / total_tests if total_tests > 0 else 0
        }
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive CUDA Evaluation Server Testing")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--mode", default="comprehensive", 
                       choices=["basic", "quick", "validation", "load", "performance", "comprehensive", "triton", "debug"],
                       help="Test mode")
    parser.add_argument("--index", type=int, help="Test specific case by index")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per test")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for validation tests")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent requests")
    parser.add_argument("--timeout", type=int, default=45, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--save-report", help="Save report to JSON file")
    parser.add_argument("--device-metrics", action="store_true", help="Enable device metrics collection")
    parser.add_argument("--kernel-types", nargs="+", default=["torch_cuda"],
                       choices=["torch", "torch_cuda", "triton", "all"],
                       help="Kernel types to test")
    parser.add_argument("--triton-opt", default="naive",
                       choices=["naive", "optimized", "both"],
                       help="Triton optimization level")
    parser.add_argument("--iocontract", action="store_true",
                       help="Enable IOContract-based Triton testing")
    
    args = parser.parse_args()
    
    # Process kernel types
    kernel_types = []
    if "all" in args.kernel_types:
        kernel_types = [KernelType.TORCH, KernelType.TORCH_CUDA, KernelType.TRITON]
    else:
        for kt in args.kernel_types:
            if kt == "torch":
                kernel_types.append(KernelType.TORCH)
            elif kt == "torch_cuda":
                kernel_types.append(KernelType.TORCH_CUDA)
            elif kt == "triton":
                kernel_types.append(KernelType.TRITON)
    
    # Configure test settings
    config = TestConfig(
        base_url=args.server,
        num_trials=args.trials,
        timeout=args.timeout,
        batch_size=args.batch_size,
        concurrent_requests=args.concurrent,
        verbose=args.verbose,
        test_mode=TestMode(args.mode),
        enable_device_metrics=args.device_metrics,
        kernel_types=kernel_types,
        triton_optimization_level=args.triton_opt,
        enable_iocontract=args.iocontract
    )
    
    # Print header
    print(f"🚀 CUDA Evaluation Server V2 - Comprehensive Test Suite")
    print(f"   Server: {config.base_url}")
    print(f"   Mode: {args.mode}")
    print(f"   Trials: {config.num_trials}")
    print(f"   Kernel Types: {', '.join(kt.value for kt in kernel_types)}")
    if KernelType.TRITON in kernel_types:
        print(f"   Triton Optimization: {args.triton_opt}")
        print(f"   IOContract Testing: {'Enabled' if args.iocontract else 'Disabled'}")
    
    # Initialize test suite
    suite = ComprehensiveTestSuite(config)
    
    # Check server health first
    success, health = suite.client.health_check()
    if not success:
        print(f"❌ Server health check failed: {health.get('error', 'Unknown error')}")
        return 1
    
    gpu_count = suite.client.get_gpu_count()
    print(f"   GPUs available: {gpu_count}")
    
    # Run tests based on mode
    try:
        if args.index is not None:
            # Test specific case
            results = suite.integration.test_specific_case(args.index, enable_device_metrics=config.enable_device_metrics)
            # Print specific case results differently
            print(f"\n{'='*60}")
            print(f"📊 Test Case {args.index} Results")
            print(f"{'='*60}")
            if "error" in results:
                print(f"❌ Error: {results['error']}")
                return 1  # Return error code
            else:
                # Check validation status
                validation_passed = results.get('validation_passed', False)
                better_than_expected = results.get('better_than_expected', False)
                validation_symbol = "✅" if validation_passed else "❌"
                
                print(f"📡 Request Status: {'✅ Success' if results.get('success', False) else '❌ Failed'}")
                print(f"⏱️ Duration: {results.get('duration', 0):.2f}s")
                if "expected" in results and "actual" in results:
                    print(f"\n📋 Validation:")
                    print(f"  Expected - Compile: {results['expected']['compile']}, Correct: {results['expected']['correct']}")
                    print(f"  Actual - Compile: {results['actual']['compile']}, Correct: {results['actual']['correct']}")
                    
                    # Show validation status with special note if better than expected
                    if better_than_expected:
                        print(f"  {validation_symbol} Validation: PASSED (⭐ Better than expected!)")
                    else:
                        print(f"  {validation_symbol} Validation: {'PASSED' if validation_passed else 'FAILED'}")
                if "performance" in results:
                    perf = results["performance"]
                    print(f"\n🚀 Performance:")
                    print(f"  Ref Runtime: {perf.get('ref_runtime_ms', 0):.3f} ms")
                    print(f"  Custom Runtime: {perf.get('custom_runtime_ms', 0):.3f} ms")
                    print(f"  Speedup: {perf.get('speedup', 0):.2f}x")
                
                # Display device metrics if available
                if "device_metrics" in results and results["device_metrics"]:
                    print(f"\n🔬 Device Metrics:")
                    format_device_metrics(results["device_metrics"])
                
                # Return appropriate exit code
                if not validation_passed:
                    print(f"\n❌ Test case {args.index} failed validation!")
                    return 1
                else:
                    print(f"\n✅ Test case {args.index} passed validation!")
                    return 0
        
        elif args.mode == "basic":
            results = suite.run_basic_tests()
            TestReporter.print_summary("Basic Tests", results, verbose=config.verbose)
        
        elif args.mode == "quick":
            # Quick test with small batch
            config.batch_size = 5
            config.concurrent_requests = 3
            results = {
                "basic": suite.run_basic_tests(),
                "validation": suite.functional.test_validation_accuracy(batch_size=5)
            }
            TestReporter.print_summary("Quick Tests", results["validation"], verbose=config.verbose)
        
        elif args.mode == "validation":
            results = suite.run_validation_tests()
            for test_name, test_results in results.items():
                TestReporter.print_summary(test_name.replace("_", " ").title(), 
                                          test_results, verbose=config.verbose)
        
        elif args.mode in ["load", "performance"]:
            results = suite.run_performance_tests()
            for test_name, test_results in results.items():
                TestReporter.print_summary(test_name.replace("_", " ").title(), 
                                          test_results, verbose=config.verbose)
        
        elif args.mode == "triton":
            # Specific mode for Triton testing
            results = suite.run_triton_tests()
            for test_name, test_results in results.items():
                if isinstance(test_results, dict):
                    TestReporter.print_summary(test_name.replace("_", " ").title(),
                                             test_results, verbose=config.verbose)
        
        elif args.mode == "comprehensive":
            results = suite.run_comprehensive_suite()
            
            # Print summaries for each category
            for category, category_results in results.items():
                if category != "overall" and isinstance(category_results, dict):
                    print(f"\n📂 {category.upper()} TESTS")
                    for test_name, test_results in category_results.items():
                        if isinstance(test_results, dict):
                            TestReporter.print_summary(
                                test_name.replace("_", " ").title(),
                                test_results,
                                verbose=config.verbose
                            )
            
            # Print overall summary
            TestReporter.print_summary("OVERALL SUMMARY", results["overall"], verbose=True)
        
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
        
        # Save report if requested
        if args.save_report:
            TestReporter.save_report(args.save_report, results)
        
        # For non-specific test modes, always successful if no exceptions
        print("\n✅ Testing complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())