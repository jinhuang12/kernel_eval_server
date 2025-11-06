"""
Unit tests for data models (RuntimeStats, DeviceMetrics, KernelMetadata, etc.)
Migrated from root test_typed_models.py
"""

import pytest
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.models import (
    RuntimeStats,
    KernelExecutionResult,
    KernelMetadata,
    DeviceMetrics,
    ComparisonDeviceMetrics,
    CompareResponse,
    EvaluationResponse
)


@pytest.mark.unit
class TestRuntimeStats:
    """Test RuntimeStats creation and serialization"""

    def test_from_dict_complete(self):
        """Test creating RuntimeStats from complete dict"""
        stats_dict = {
            "mean": 1.5,
            "std": 0.2,
            "min": 1.2,
            "max": 1.8,
            "median": 1.5,
            "percentile_95": 1.7,
            "percentile_99": 1.75
        }

        stats = RuntimeStats.from_dict(stats_dict)

        assert stats.mean == 1.5
        assert stats.std == 0.2
        assert stats.min == 1.2
        assert stats.max == 1.8
        assert stats.median == 1.5
        assert stats.percentile_95 == 1.7
        assert stats.percentile_99 == 1.75

    def test_from_dict_partial(self):
        """Test RuntimeStats with missing fields defaults to 0.0"""
        partial_dict = {"mean": 2.0, "std": 0.3}
        stats = RuntimeStats.from_dict(partial_dict)

        assert stats.mean == 2.0
        assert stats.std == 0.3
        assert stats.min == 0.0  # Should default to 0.0
        assert stats.max == 0.0
        assert stats.median == 0.0

    def test_to_dict(self):
        """Test converting RuntimeStats to dict"""
        stats = RuntimeStats(mean=1.5, std=0.2, min=1.2, max=1.8, median=1.5)
        stats_dict = {
            "mean": stats.mean,
            "std": stats.std,
            "min": stats.min,
            "max": stats.max,
            "median": stats.median,
            "percentile_95": stats.percentile_95,
            "percentile_99": stats.percentile_99
        }

        assert stats_dict["mean"] == 1.5
        assert stats_dict["std"] == 0.2


@pytest.mark.unit
class TestDeviceMetrics:
    """Test DeviceMetrics creation and serialization"""

    def test_from_dict_nested_structure(self):
        """Test creating DeviceMetrics from NCU parser output"""
        ncu_output = {
            "speed_of_light": {
                "compute_throughput_pct": 85.5,
                "memory_throughput_pct": 72.3,
                "sm_throughput_pct": 80.1
            },
            "detailed_metrics": {
                "l1_hit_rate_pct": 95.2,
                "l2_hit_rate_pct": 88.7,
                "warp_occupancy_pct": 75.0
            },
            "memory_metrics": {
                "dram_avg_bandwidth_gb_s": 450.0,
                "l2_throughput_pct": 68.5
            }
        }

        metrics = DeviceMetrics.from_dict(ncu_output)
        metrics_dict = metrics.to_dict()

        # Verify structure preservation
        assert "speed_of_light" in metrics_dict
        assert "detailed_metrics" in metrics_dict
        assert "memory_metrics" in metrics_dict
        assert metrics_dict["speed_of_light"]["compute_throughput_pct"] == 85.5
        assert metrics_dict["detailed_metrics"]["l1_hit_rate_pct"] == 95.2

    def test_from_dict_empty(self):
        """Test handling empty DeviceMetrics"""
        empty_metrics = DeviceMetrics.from_dict({})
        empty_dict = empty_metrics.to_dict()

        assert len(empty_dict) == 0

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values"""
        metrics = DeviceMetrics.from_dict({
            "speed_of_light": {"compute_throughput_pct": 90.0}
        })
        metrics_dict = metrics.to_dict()

        # Should only contain categories that have data
        assert "speed_of_light" in metrics_dict
        assert len(metrics_dict) == 1

    def test_to_dict_rounds_floats_to_3_decimals(self):
        """Test that to_dict rounds float values to 3 decimal places"""
        metrics = DeviceMetrics.from_dict({
            "speed_of_light": {
                "compute_throughput_pct": 85.123456789,
                "memory_throughput_pct": 72.987654321,
                "sm_throughput_pct": 80.111111111
            },
            "detailed_metrics": {
                "l1_hit_rate_pct": 95.55555555,
                "l2_hit_rate_pct": 88.99999999,
                "warp_occupancy_pct": 75.0
            },
            "memory_metrics": {
                "dram_avg_bandwidth_gb_s": 450.123456789,
                "l2_throughput_pct": 68.5
            }
        })
        metrics_dict = metrics.to_dict()

        # Verify all float values are rounded to 3 decimal places
        assert metrics_dict["speed_of_light"]["compute_throughput_pct"] == 85.123
        assert metrics_dict["speed_of_light"]["memory_throughput_pct"] == 72.988
        assert metrics_dict["speed_of_light"]["sm_throughput_pct"] == 80.111
        assert metrics_dict["detailed_metrics"]["l1_hit_rate_pct"] == 95.556
        assert metrics_dict["detailed_metrics"]["l2_hit_rate_pct"] == 89.0
        assert metrics_dict["detailed_metrics"]["warp_occupancy_pct"] == 75.0
        assert metrics_dict["memory_metrics"]["dram_avg_bandwidth_gb_s"] == 450.123
        assert metrics_dict["memory_metrics"]["l2_throughput_pct"] == 68.5


@pytest.mark.unit
class TestComparisonDeviceMetrics:
    """Test ComparisonDeviceMetrics creation and serialization"""

    def test_to_dict_rounds_floats_to_3_decimals(self):
        """Test that ComparisonDeviceMetrics rounds float values to 3 decimal places"""
        original_metrics = DeviceMetrics.from_dict({
            "speed_of_light": {
                "compute_throughput_pct": 85.123456789,
                "memory_throughput_pct": 72.987654321
            }
        })

        custom_metrics = DeviceMetrics.from_dict({
            "speed_of_light": {
                "compute_throughput_pct": 90.555555555,
                "memory_throughput_pct": 80.999999999
            }
        })

        comparison = ComparisonDeviceMetrics(
            original_device_metrics=original_metrics,
            custom_device_metrics=custom_metrics
        )

        comparison_dict = comparison.to_dict()

        # Verify original metrics are rounded
        assert comparison_dict["original_device_metrics"]["speed_of_light"]["compute_throughput_pct"] == 85.123
        assert comparison_dict["original_device_metrics"]["speed_of_light"]["memory_throughput_pct"] == 72.988

        # Verify custom metrics are rounded
        assert comparison_dict["custom_device_metrics"]["speed_of_light"]["compute_throughput_pct"] == 90.556
        assert comparison_dict["custom_device_metrics"]["speed_of_light"]["memory_throughput_pct"] == 81.0

    def test_from_dict_and_to_dict_roundtrip(self):
        """Test that ComparisonDeviceMetrics can roundtrip through dict conversion"""
        test_data = {
            "original_device_metrics": {
                "speed_of_light": {"compute_throughput_pct": 85.5}
            },
            "custom_device_metrics": {
                "speed_of_light": {"compute_throughput_pct": 90.5}
            }
        }

        comparison = ComparisonDeviceMetrics.from_dict(test_data)
        comparison_dict = comparison.to_dict()

        assert "original_device_metrics" in comparison_dict
        assert "custom_device_metrics" in comparison_dict
        assert comparison_dict["original_device_metrics"]["speed_of_light"]["compute_throughput_pct"] == 85.5


@pytest.mark.unit
class TestKernelMetadata:
    """Test KernelMetadata creation and serialization"""

    def test_with_device_metrics(self):
        """Test metadata with device metrics"""
        device_metrics = DeviceMetrics.from_dict({
            "speed_of_light": {"compute_throughput_pct": 90.0}
        })

        metadata = KernelMetadata(
            gpu_id=0,
            device_metrics=device_metrics,
            kernel_name="matmul_kernel",
            kernel_type="triton",
            gpu_type="h100"
        )

        metadata_dict = metadata.to_dict()

        assert metadata_dict["gpu_id"] == 0
        assert metadata_dict["kernel_name"] == "matmul_kernel"
        assert metadata_dict["kernel_type"] == "triton"
        assert "device_metrics" in metadata_dict

    def test_minimal_metadata(self):
        """Test metadata with only required fields"""
        metadata = KernelMetadata(gpu_id=1)
        metadata_dict = metadata.to_dict()

        assert metadata_dict["gpu_id"] == 1
        assert "kernel_name" not in metadata_dict or metadata_dict["kernel_name"] is None


@pytest.mark.unit
class TestKernelExecutionResult:
    """Test KernelExecutionResult creation and serialization"""

    def test_successful_execution(self):
        """Test successful kernel execution result"""
        metadata = KernelMetadata(gpu_id=0, kernel_name="test_kernel")
        runtime_stats = RuntimeStats(mean=1.5, std=0.2, min=1.2, max=1.8, median=1.5)

        result = KernelExecutionResult(
            compiled=True,
            correctness=True,
            runtime=1.5,
            metadata=metadata,
            runtime_stats=runtime_stats
        )

        result_dict = result.to_dict()

        assert result_dict["compiled"] == True
        assert result_dict["correctness"] == True
        assert result_dict["runtime"] == 1.5
        assert "runtime_stats" in result_dict

    def test_compilation_failure(self):
        """Test failed compilation result"""
        metadata = KernelMetadata(gpu_id=1)

        result = KernelExecutionResult(
            compiled=False,
            correctness=False,
            runtime=0.0,
            metadata=metadata,
            compilation_error="Syntax error in kernel code"
        )

        result_dict = result.to_dict()

        assert result_dict["compiled"] == False
        assert "compilation_error" in result_dict
        assert result_dict["compilation_error"] == "Syntax error in kernel code"

    def test_validation_failure(self):
        """Test validation failure result"""
        metadata = KernelMetadata(gpu_id=0)
        runtime_stats = RuntimeStats(mean=1.5, std=0.2, min=1.2, max=1.8, median=1.5)

        result = KernelExecutionResult(
            compiled=True,
            correctness=False,
            runtime=1.5,
            metadata=metadata,
            runtime_stats=runtime_stats,
            validation_error="Output mismatch"
        )

        result_dict = result.to_dict()

        assert result_dict["compiled"] == True
        assert result_dict["correctness"] == False
        assert "validation_error" in result_dict


@pytest.mark.unit
class TestResponseModels:
    """Test the full response models (CompareResponse, EvaluationResponse)"""

    def test_compare_response(self):
        """Test CompareResponse serialization"""
        kernel_exec_result = KernelExecutionResult(
            compiled=True,
            correctness=True,
            runtime=1.5,
            metadata=KernelMetadata(gpu_id=0),
            runtime_stats=RuntimeStats(mean=1.5, std=0.2, min=1.2, max=1.8, median=1.5)
        )

        ref_runtime = RuntimeStats(mean=2.0, std=0.3, min=1.7, max=2.3, median=2.0)

        compare_response = CompareResponse(
            job_id="test-123",
            kernel_exec_result=kernel_exec_result,
            ref_runtime=ref_runtime,
            pod_name="cuda-eval-pod",
            pod_ip="10.0.0.1",
            status="success"
        )

        # Convert to dict (as would happen in FastAPI)
        response_dict = compare_response.model_dump()

        assert response_dict["job_id"] == "test-123"
        assert response_dict["status"] == "success"

        # Manually convert dataclasses (as done in app.py)
        if hasattr(compare_response.kernel_exec_result, 'to_dict'):
            response_dict['kernel_exec_result'] = compare_response.kernel_exec_result.to_dict()

        if hasattr(compare_response, 'ref_runtime') and compare_response.ref_runtime:
            response_dict['ref_runtime'] = {
                "mean": compare_response.ref_runtime.mean,
                "std": compare_response.ref_runtime.std,
                "min": compare_response.ref_runtime.min,
                "max": compare_response.ref_runtime.max,
                "median": compare_response.ref_runtime.median,
                "percentile_95": compare_response.ref_runtime.percentile_95,
                "percentile_99": compare_response.ref_runtime.percentile_99
            }

        # Verify JSON serializable
        json_str = json.dumps(response_dict, indent=2)
        assert len(json_str) > 0

    def test_evaluation_response(self):
        """Test EvaluationResponse serialization"""
        kernel_exec_result = KernelExecutionResult(
            compiled=True,
            correctness=True,
            runtime=1.5,
            metadata=KernelMetadata(gpu_id=0),
            runtime_stats=RuntimeStats(mean=1.5, std=0.2, min=1.2, max=1.8, median=1.5)
        )

        eval_response = EvaluationResponse(
            job_id="eval-456",
            kernel_exec_result=kernel_exec_result,
            pod_name="cuda-eval-pod",
            pod_ip="10.0.0.1",
            status="success"
        )

        eval_dict = eval_response.model_dump()

        if hasattr(eval_response.kernel_exec_result, 'to_dict'):
            eval_dict['kernel_exec_result'] = eval_response.kernel_exec_result.to_dict()

        # Verify JSON serializable
        eval_json = json.dumps(eval_dict, indent=2)
        assert len(eval_json) > 0
        assert eval_dict["job_id"] == "eval-456"


@pytest.mark.unit
class TestModelIntegration:
    """Test integration between different model types"""

    def test_full_response_chain(self):
        """Test complete response chain with all models"""
        # Create device metrics
        device_metrics = DeviceMetrics.from_dict({
            "speed_of_light": {
                "compute_throughput_pct": 85.5,
                "memory_throughput_pct": 72.3
            }
        })

        # Create metadata
        metadata = KernelMetadata(
            gpu_id=0,
            device_metrics=device_metrics,
            kernel_name="test_kernel",
            kernel_type="triton"
        )

        # Create runtime stats
        runtime_stats = RuntimeStats(
            mean=1.5,
            std=0.2,
            min=1.2,
            max=1.8,
            median=1.5,
            percentile_95=1.7,
            percentile_99=1.75
        )

        # Create execution result
        exec_result = KernelExecutionResult(
            compiled=True,
            correctness=True,
            runtime=1.5,
            metadata=metadata,
            runtime_stats=runtime_stats
        )

        # Verify everything is serializable
        result_dict = exec_result.to_dict()

        assert result_dict["compiled"] == True
        assert result_dict["correctness"] == True
        assert "metadata" in result_dict
        assert "runtime_stats" in result_dict
        assert "device_metrics" in result_dict["metadata"]

        # Verify percentiles are present
        assert "percentile_95" in result_dict["runtime_stats"]
        assert "percentile_99" in result_dict["runtime_stats"]
