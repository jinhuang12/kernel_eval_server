# Base image with CUDA 12.6, cuDNN, and development tools
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1
ENV CUPY_KERNEL_CACHE_DIR=/tmp/cupy_kernel_cache
ENV ENABLE_DEVICE_METRICS=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symlink for ninja if needed
RUN ln -sf /usr/bin/ninja /usr/local/bin/ninja

# Upgrade pip (already installed via get-pip.py)
RUN python3.12 -m pip install --upgrade setuptools wheel

# Install NVIDIA Nsight Compute (NCU)
# Download and install the latest NCU version compatible with CUDA 12.6
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-compute-2024.3.2_2024.3.2.3-1_amd64.deb && \
    dpkg -i nsight-compute-2024.3.2_2024.3.2.3-1_amd64.deb && \
    rm nsight-compute-2024.3.2_2024.3.2.3-1_amd64.deb

# Set NCU environment variables
ENV NCU_HOME=/opt/nvidia/nsight-compute/2024.3.2
ENV PATH=${NCU_HOME}:${PATH}

# Add NCU Python module to PYTHONPATH
ENV PYTHONPATH=${NCU_HOME}/extras/python:${PYTHONPATH}

# Add KernelBench to Python path for imports
ENV PYTHONPATH=/app:${PYTHONPATH}

# Create working directory
WORKDIR /app

# Copy both requirements files
COPY requirements.txt /app/requirements.main.txt
COPY requirements.eval_server_v2.txt /app/requirements.eval_server.txt

# Install Python dependencies from both requirements files
# Install main KernelBench requirements first
RUN pip3 install --no-cache-dir -r requirements.main.txt

# Then install eval server specific requirements
RUN pip3 install --no-cache-dir -r requirements.eval_server.txt

# Copy the entire KernelBench package
COPY KernelBench /app/KernelBench

# Set the working directory to the eval server
WORKDIR /app/KernelBench/scripts/cuda_eval_server_v2

# Install client package for testing (optional)
RUN pip3 install -e /app/KernelBench/scripts/cuda_eval_server_v2/client || true

# Create cache directories with proper permissions
RUN mkdir -p /tmp/cupy_kernel_cache \
    && mkdir -p /tmp/torch_extensions \
    && mkdir -p /root/.cache \
    && chmod -R 777 /tmp/cupy_kernel_cache \
    && chmod -R 777 /tmp/torch_extensions \
    && chmod -R 777 /root/.cache

# Expose the default port
EXPOSE 8000

# Default command
CMD ["python3", "main.py", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]