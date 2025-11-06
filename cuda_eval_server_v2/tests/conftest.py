"""
Pytest configuration and shared fixtures for CUDA Evaluation Server tests
"""

import os
import sys
import pytest
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.test_data_loader import get_loader

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Pytest markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests that don't require server")
    config.addinivalue_line("markers", "integration: Integration tests requiring server")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests that take > 10 seconds")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "triton: Tests requiring Triton")
    config.addinivalue_line("markers", "cuda: Tests requiring CUDA")

# Common fixtures
@pytest.fixture
def server_url():
    """Get server URL from environment or use default"""
    return os.getenv("CUDA_EVAL_SERVER_URL", "http://localhost:8000")

@pytest.fixture
def test_index():
    """Get test index from environment or use default"""
    return os.getenv("TEST_INDEX", "test-index")

# Test data fixtures
@pytest.fixture
def test_data_loader():
    """Get test data loader instance"""
    return get_loader()

@pytest.fixture
def cuda_test_cases(test_data_loader) -> List[Tuple[str, Dict[str, Any]]]:
    """Get all CUDA test cases"""
    return test_data_loader.get_cuda_test_cases()

@pytest.fixture
def triton_test_cases(test_data_loader) -> List[Tuple[str, Dict[str, Any]]]:
    """Get all Triton test cases"""
    return test_data_loader.get_triton_test_cases()

@pytest.fixture
def all_test_cases(test_data_loader) -> List[Tuple[str, Dict[str, Any]]]:
    """Get all available test cases"""
    return test_data_loader.get_all_test_cases()

def pytest_generate_tests(metafunc):
    """Generate parametrized tests from test data"""
    loader = get_loader()
    
    # Parametrize CUDA test cases
    if "cuda_test_case" in metafunc.fixturenames:
        test_cases = loader.get_cuda_test_cases()
        metafunc.parametrize("cuda_test_case", 
                           [tc for _, tc in test_cases],
                           ids=[name for name, _ in test_cases])
    
    # Parametrize Triton test cases
    if "triton_test_case" in metafunc.fixturenames:
        test_cases = loader.get_triton_test_cases()
        metafunc.parametrize("triton_test_case",
                           [tc for _, tc in test_cases],
                           ids=[name for name, _ in test_cases])
    
    # Parametrize all test cases
    if "test_case_data" in metafunc.fixturenames:
        test_cases = loader.get_all_test_cases()
        metafunc.parametrize("test_case_data",
                           [tc for _, tc in test_cases],
                           ids=[name for name, _ in test_cases])