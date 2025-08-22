#!/usr/bin/env python3
"""Triton integration tests for the CUDA evaluation server.

This script mirrors the CLI of ``test_server_torch_cuda.py`` and exercises the
``/evaluate`` endpoint using Triton kernels. Test cases are stored as JSON
payloads under ``test_data/triton`` with expected outcomes. The script posts each
payload to the server and asserts on the response according to the expectations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import requests

# ---------------------------------------------------------------------------
# Test case loading and execution
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data" / "triton"


def load_test_cases() -> List[Dict[str, Any]]:
    """Load all JSON test cases from ``test_data/triton``."""
    cases: List[Dict[str, Any]] = []
    if not TEST_DATA_DIR.exists():
        return cases
    for path in sorted(TEST_DATA_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
            case["path"] = str(path)
            cases.append(case)
    return cases


def post_evaluation(server: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{server.rstrip('/')}/evaluate"
    timeout = payload.get("timeout", 120) + 30
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_case(server: str, case: Dict[str, Any]) -> None:
    payload = case["request"]
    expected = case.get("expected", {})

    result = post_evaluation(server, payload)

    assert result["status"] == expected.get("status", "success"), result

    ker = result.get("kernel_exec_result", {})
    if "is_correct" in expected:
        is_correct = ker.get("is_correct", ker.get("correctness"))
        assert is_correct is expected["is_correct"], ker

    metadata = ker.get("metadata", {})
    if "min_speedup" in expected:
        speedup = metadata.get("speedup", 0.0)
        assert speedup >= expected["min_speedup"], f"Expected speedup ≥ {expected['min_speedup']}, got {speedup}"

    if expected.get("expect_device_metrics"):
        metrics = metadata.get("device_metrics", {})
        assert "original_device_metrics" in metrics and "custom_device_metrics" in metrics, metrics

    print(json.dumps(result, indent=2))
    print(f"Test '{case.get('name', case.get('path'))}' passed ✓")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Triton server integration tests")
    parser.add_argument("--server", default="http://localhost:8000", help="Base server URL")
    parser.add_argument("--basic", action="store_true", help="Run the first Triton test case")
    parser.add_argument("--test-index", type=int, default=None, help="Index of test case to run")
    args = parser.parse_args()

    cases = load_test_cases()
    if not cases:
        raise RuntimeError(f"No test cases found in {TEST_DATA_DIR}")

    index = 0 if args.basic or args.test_index is None else args.test_index
    if index < 0 or index >= len(cases):
        raise IndexError(f"test-index {index} out of range (found {len(cases)} cases)")

    run_case(args.server, cases[index])


if __name__ == "__main__":  # pragma: no cover
    main()
