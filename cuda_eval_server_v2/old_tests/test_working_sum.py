#!/usr/bin/env python3
"""Test working sum_reduction directly"""

import requests
import json

# Use the working test data file
with open('test_data/triton/sum_reduction_io.json', 'r') as f:
    request_data = json.load(f)

# Set number of trials
request_data['num_trials'] = 5

print("Sending request...")
response = requests.post(
    'http://localhost:8000/',
    json=request_data,
    timeout=30
)

print(f"Status code: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'success':
        print("✅ Test passed!")
    else:
        print(f"Error: {result.get('error')}")
    print(f'{result=}')
else:
    print(f"HTTP Error: {response.text[:500]}")