#!/usr/bin/env python3
"""
Integration test for CUDA backend through the eval server HTTP API
Usage: python test_server_cuda.py [server_url]
"""

import sys
import json
import requests
import time
import os
import glob
from typing import Dict, Any, List, Tuple

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:8000"

# Default test data directory
DEFAULT_TEST_DATA_DIR = "test_data/cuda"


def load_json_test_case(file_path: str) -> Dict[str, Any]:
    """Load a JSON test case from file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load test case from {file_path}: {e}")
        return None


def get_test_cases(test_data_dir: str = None, specific_files: List[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """Get all test cases from JSON files or specific files"""
    if test_data_dir is None:
        test_data_dir = DEFAULT_TEST_DATA_DIR
    
    test_cases = []
    
    if specific_files:
        # Load specific files
        for file_path in specific_files:
            if not os.path.exists(file_path):
                print(f"⚠️ Test file not found: {file_path}")
                continue
            
            test_case = load_json_test_case(file_path)
            if test_case:
                # Extract name from filename
                name = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
                test_cases.append((name, test_case))
    else:
        # Load all JSON files from test data directory
        if not os.path.exists(test_data_dir):
            print(f"⚠️ Test data directory not found: {test_data_dir}")
            return test_cases
        
        json_files = glob.glob(os.path.join(test_data_dir, "*.json"))
        json_files.sort()  # Ensure consistent order
        
        for file_path in json_files:
            test_case = load_json_test_case(file_path)
            if test_case:
                # Extract name from filename
                name = os.path.splitext(os.path.basename(file_path))[0].replace('_', ' ').title()
                test_cases.append((name, test_case))
    
    return test_cases


def test_kernel(server_url: str, test_case: Dict[str, Any], kernel_name: str, endpoint: str = "compare"):
    """Test a CUDA kernel through the server"""
    
    print(f"\n{'='*60}")
    print(f"Testing {kernel_name} (endpoint: /{endpoint})")
    print('='*60)
    
    try:
        # Prepare request based on endpoint
        if endpoint == "evaluate":
            # Extract custom_kernel for single kernel evaluation
            if "custom_kernel" not in test_case:
                print(f"❌ No custom_kernel found in test case for evaluate endpoint")
                return False
            
            request = {
                "kernel": test_case["custom_kernel"],
                "num_trials": test_case.get("num_trials", 100),
                "timeout": test_case.get("timeout", 30)
            }
            endpoint_url = f"{server_url}/evaluate"
        else:
            # Use full compare request
            request = test_case
            endpoint_url = f"{server_url}/compare"
        
        # Send request to server
        print(f"Sending request to {endpoint_url}...")
        response = requests.post(
            endpoint_url,
            json=request,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"   Job ID: {result.get('job_id', 'N/A')}")
            
            if endpoint == "compare":
                # Handle compare endpoint response
                ref_result = result.get('ref_runtime', {})
                custom_result = result.get('kernel_exec_result', {})
                
                # Check reference kernel
                if ref_result.get('compiled'):
                    print(f"✅ Reference kernel compiled successfully")
                else:
                    print(f"❌ Reference compilation failed: {ref_result.get('compilation_error', 'Unknown error')}")
                
                # Check custom kernel
                if custom_result.get('compiled'):
                    print(f"✅ Custom kernel compiled successfully")
                else:
                    print(f"❌ Custom compilation failed: {custom_result.get('compilation_error', 'Unknown error')}")
                    return False
                
                # Show comparison results
                if result.get('is_correct'):
                    print(f"✅ Correctness validation passed")
                else:
                    print(f"❌ Correctness validation failed")
                
                if result.get('speedup'):
                    speedup = result.get('speedup', 0)
                    print(f"📊 Speedup: {speedup:.2f}x")
                
                # Show runtime stats for custom kernel
                if custom_result.get('runtime_stats'):
                    runtime = custom_result.get('runtime_stats', {})
                    print(f"⏱️ Custom kernel performance:")
                    print(f"   Mean runtime: {runtime.get('mean', 0)*1000:.3f} ms")
                    print(f"   Min runtime: {runtime.get('min', 0)*1000:.3f} ms")
                    print(f"   Max runtime: {runtime.get('max', 0)*1000:.3f} ms")
                    
            else:
                # Handle evaluate endpoint response (single kernel)
                exec_result = result.get('kernel_exec_result', {})
                
                # Check compilation
                if exec_result.get('compiled'):
                    print(f"✅ Kernel compiled successfully")
                else:
                    print(f"❌ Compilation failed: {exec_result.get('compilation_error', 'Unknown error')}")
                    return False
                
                # Check profiling
                if exec_result.get('runtime_stats'):
                    runtime = exec_result.get('runtime_stats', {})
                    print(f"✅ Profiling successful")
                    print(f"   Mean runtime: {runtime.get('mean', 0)*1000:.3f} ms")
                    print(f"   Min runtime: {runtime.get('min', 0)*1000:.3f} ms")
                    print(f"   Max runtime: {runtime.get('max', 0)*1000:.3f} ms")
                else:
                    print(f"⚠️ Profiling failed: {exec_result.get('profiling_error', 'Unknown error')}")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_server_health(server_url: str) -> bool:
    """Check if server is healthy and CUDA backend is available"""
    
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if not response.status_code == 200:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False


def main():
    """Main test function"""
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python test_server_cuda.py [options] [server_url] [test_file1.json] [test_file2.json] ...")
        print("")
        print("Options:")
        print("  --endpoint ENDPOINT    API endpoint to test: 'compare' or 'evaluate' (default: compare)")
        print("")
        print("Arguments:")
        print(f"  server_url    Server URL (default: {DEFAULT_SERVER_URL})")
        print("  test_files    Specific JSON test files to run (default: all files in test_data/cuda/)")
        print("")
        print("Examples:")
        print("  python test_server_cuda.py                                         # Run all tests against localhost:8000 (compare)")
        print("  python test_server_cuda.py --endpoint evaluate                     # Run all tests using evaluate endpoint")
        print("  python test_server_cuda.py http://remote:8000                      # Run all tests against remote server (compare)")
        print("  python test_server_cuda.py --endpoint evaluate http://localhost:8000 vector_add.json  # Run specific test on evaluate endpoint")
        return
    
    # Parse endpoint option
    endpoint = "compare"  # default
    args = sys.argv[1:]
    
    if args and args[0] == "--endpoint":
        if len(args) < 2:
            print("Error: --endpoint requires a value (compare or evaluate)")
            return
        endpoint = args[1]
        if endpoint not in ["compare", "evaluate"]:
            print(f"Error: Invalid endpoint '{endpoint}'. Must be 'compare' or 'evaluate'")
            return
        args = args[2:]  # Remove endpoint args
    
    # Get server URL from remaining arguments or use default
    server_url = args[0] if len(args) > 0 else DEFAULT_SERVER_URL
    
    print("="*60)
    print("CUDA Backend Server Integration Test")
    print(f"Server: {server_url}")
    print(f"Endpoint: /{endpoint}")
    print("="*60)
    
    # Check server health
    print("\nChecking server health...")
    if not check_server_health(server_url):
        print("\n⚠️ Server health check failed. Make sure:")
        print("  1. The server is running (python main.py)")
        print("  2. CUDA backend is registered")
        return
    
    # Get test cases from JSON files or command line arguments
    test_files = None
    if len(args) > 1:
        # Additional arguments are treated as test file paths
        test_files = args[1:]
        print(f"Running specific test files: {', '.join(test_files)}")
    
    test_cases = get_test_cases(specific_files=test_files)
    
    if not test_cases:
        print("\n⚠️ No test cases found. Make sure:")
        print("  1. The test_data/cuda directory exists")
        print("  2. JSON test files are present in the directory")
        print("  3. JSON files are properly formatted")
        return
    
    results = []
    for name, test_case in test_cases:
        success = test_kernel(server_url, test_case, name, endpoint)
        results.append((name, success))
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    successful = sum(1 for _, success in results if success)
    print(f"✅ Successful: {successful}/{len(test_cases)}")
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")


if __name__ == "__main__":
    main()