"""
Setup script for Kernel Evaluation Client Library
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


setup(
    name="kernel-eval-client",
    version="0.1.0",
    author="KernelBench Team",
    description="Python client library for Kernel Evaluation Server API",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/KernelBench",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
        "examples": [
            "numpy>=1.20.0",  # For advanced examples
        ]
    },
    keywords="kernel evaluation gpu cuda triton pytorch performance profiling",
    project_urls={
        "Documentation": "https://github.com/your-org/KernelBench/tree/main/KernelBench/scripts/cuda_eval_server_v2/client",
        "Source": "https://github.com/your-org/KernelBench",
        "Issues": "https://github.com/your-org/KernelBench/issues",
    },
)