markdown# Fast DDS with GPU/CUDA Integration - Complete Documentation

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Docker Backup & Recovery](#docker-backup--recovery)
- [Performance Results](#performance-results)
- [Troubleshooting](#troubleshooting)

## Project Overview

High-performance data distribution system using eProsima Fast DDS with NVIDIA CUDA GPU acceleration. Implements publisher-subscriber pattern for GPU-processed data distribution.

### Key Features
- 4 concurrent CUDA streams
- 4 GPU kernels (Vector Add, FFT, Convolution, Reduction)
- Fast DDS with UDP transport
- Docker containerization with GPU support
- Comprehensive benchmarking tools

## Installation

### Prerequisites
- NVIDIA GPU (Compute Capability 7.0+)
- Docker Engine
- NVIDIA Container Toolkit
- 8GB+ RAM

### Step 1: Install NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
Step 2: Clone Repository
bashgit clone https://github.com/yourusername/fastdds-gpu-project.git
cd fastdds-gpu-project
Step 3: Create Project Structure
bash# Create all necessary files
mkdir -p src idl benchmark_results

# Save this script as setup.sh and run it
cat > setup.sh << 'SCRIPT'
#!/bin/bash

# Create CMakeLists.txt
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.18)
project(FastDDS_GPU_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(fastcdr REQUIRED)
find_package(fastrtps REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
include_directories(${CUDA_INCLUDE_DIRS})

# Add your executables here
add_executable(fixed_transport_publisher
    src/fixed_transport_publisher.cpp
)
target_link_libraries(fixed_transport_publisher
    fastrtps
    fastcdr
    pthread
)

add_executable(fixed_transport_subscriber
    src/fixed_transport_subscriber.cpp
)
target_link_libraries(fixed_transport_subscriber
    fastrtps
    fastcdr
    pthread
)
EOF

echo "Setup complete!"
SCRIPT

chmod +x setup.sh
./setup.sh
Step 4: Build Docker Image
bash# Build the image
docker build -t fastdds:gpu .

# Verify build
docker images | grep fastdds
Usage Guide
Basic Test
bash# Terminal 1: Start subscriber
docker run --rm --network host fastdds:gpu \
    /opt/fastdds-gpu/build/fixed_transport_subscriber

# Terminal 2: Start publisher
docker run --rm --gpus all --network host fastdds:gpu \
    /opt/fastdds-gpu/build/fixed_transport_publisher
Run GPU Benchmark
bash# Create results directory
mkdir -p benchmark_results

# Run benchmark
docker run --rm --gpus all --network host \
    -v $(pwd)/benchmark_results:/results \
    fastdds:gpu /opt/fastdds-gpu/build/gpu_parallel_benchmark
View Results
bash# Check benchmark output
cat benchmark_results/benchmark_results.csv

# Monitor GPU usage
nvidia-smi -l 1
Docker Backup & Recovery
Creating Backups
Method 1: Save Docker Image to TAR File
bash# Save the Docker image
docker save -o fastdds-gpu-backup.tar fastdds:gpu

# Compress the backup (optional but recommended)
gzip fastdds-gpu-backup.tar

# Verify backup size
ls -lh fastdds-gpu-backup.tar.gz
Method 2: Push to Docker Registry
bash# Tag for registry
docker tag fastdds:gpu yourusername/fastdds-gpu:v1.0

# Login to Docker Hub
docker login

# Push to registry
docker push yourusername/fastdds-gpu:v1.0
Method 3: Export Running Container
bash# If you have a running container with modifications
docker export container_name > fastdds-gpu-container.tar
Backup Project Files
bash# Create complete backup including source and results
tar -czf fastdds-project-backup-$(date +%Y%m%d).tar.gz \
    src/ \
    idl/ \
    CMakeLists.txt \
    Dockerfile \
    docker-compose.yml \
    benchmark_results/

# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup Docker image
echo "Backing up Docker image..."
docker save -o $BACKUP_DIR/fastdds-gpu-image-$DATE.tar fastdds:gpu

# Backup project files
echo "Backing up project files..."
tar -czf $BACKUP_DIR/fastdds-project-$DATE.tar.gz \
    src/ idl/ CMakeLists.txt Dockerfile docker-compose.yml \
    benchmark_results/ 2>/dev/null

# Compress Docker image
echo "Compressing Docker image..."
gzip $BACKUP_DIR/fastdds-gpu-image-$DATE.tar

echo "Backup complete!"
echo "Files saved in $BACKUP_DIR/"
ls -lh $BACKUP_DIR/
EOF

chmod +x backup.sh
./backup.sh
Recovery Instructions
Method 1: Restore from TAR File
bash# Decompress if needed
gunzip fastdds-gpu-backup.tar.gz

# Load Docker image
docker load -i fastdds-gpu-backup.tar

# Verify image loaded
docker images | grep fastdds
Method 2: Pull from Registry
bash# Pull from Docker Hub
docker pull yourusername/fastdds-gpu:v1.0

# Tag for local use
docker tag yourusername/fastdds-gpu:v1.0 fastdds:gpu
Method 3: Complete Project Recovery
bash# Extract project backup
tar -xzf fastdds-project-backup-20240902.tar.gz

# Load Docker image
docker load -i fastdds-gpu-backup.tar

# Rebuild if necessary
docker build -t fastdds:gpu .

# Verify functionality
docker run --rm --gpus all fastdds:gpu nvidia-smi
Automated Recovery Script
bashcat > restore.sh << 'EOF'
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./restore.sh <backup-date>"
    echo "Example: ./restore.sh 20240902"
    exit 1
fi

BACKUP_DATE=$1
BACKUP_DIR="backups"

# Check if backup exists
if [ ! -f "$BACKUP_DIR/fastdds-gpu-image-$BACKUP_DATE.tar.gz" ]; then
    echo "Backup not found: $BACKUP_DIR/fastdds-gpu-image-$BACKUP_DATE.tar.gz"
    exit 1
fi

echo "Restoring from backup $BACKUP_DATE..."

# Restore Docker image
echo "Loading Docker image..."
gunzip -c $BACKUP_DIR/fastdds-gpu-image-$BACKUP_DATE.tar.gz | docker load

# Restore project files
if [ -f "$BACKUP_DIR/fastdds-project-$BACKUP_DATE.tar.gz" ]; then
    echo "Restoring project files..."
    tar -xzf $BACKUP_DIR/fastdds-project-$BACKUP_DATE.tar.gz
fi

# Verify restoration
echo "Verification:"
docker images | grep fastdds
echo ""
echo "Restoration complete!"
echo "Test with: docker run --rm --gpus all fastdds:gpu nvidia-smi"
EOF

chmod +x restore.sh

# Use it like this:
./restore.sh 20240902_143022
Performance Results
Benchmark Metrics
Execution ModeTime (ms)Throughput (GB/s)SpeedupSequential120.5133.21.0xParallel45.3354.12.66xOverlapped52.1307.82.31x
GPU Utilization (RTX 4060)

Vector Addition: 0.045 ms
FFT Transform: 0.074 ms
1D Convolution: 0.042 ms
Parallel Reduction: 0.041 ms

Troubleshooting
Docker Issues
GPU Not Detected
bash# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# If fails, reinstall nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
Network Issues Between Containers
bash# Use host network
docker run --network host ...

# Or create custom network
docker network create dds-net
docker run --network dds-net ...
Permission Denied
bash# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
Build Issues
CMake Cannot Find Fast DDS
bash# Inside Dockerfile, ensure paths are correct
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
CUDA Compilation Errors
bash# Check CUDA version compatibility
nvcc --version
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvcc --version
Runtime Issues
No Messages Received
bash# Check if using correct transport
# In your code, ensure UDP transport:
pqos.transport().use_builtin_transports = false;
auto udp_transport = std::make_shared<UDPv4TransportDescriptor>();
Out of GPU Memory
bash# Reduce data size in code
static const int DATA_SIZE = 524288;  // Reduce from 1048576

# Or limit GPU memory
docker run --gpus '"device=0,capabilities=compute,utility,memory=2000"' ...
Maintenance
Update Docker Image
bash# Rebuild with latest changes
docker build --no-cache -t fastdds:gpu .

# Tag with version
docker tag fastdds:gpu fastdds:gpu-v1.1

# Keep old version
docker tag fastdds:gpu fastdds:gpu-backup
Clean Up
bash# Remove unused images
docker image prune

# Remove all stopped containers
docker container prune

# Full cleanup (careful!)
docker system prune -a
Version Control
bash# Save image with version tag
docker save -o fastdds-gpu-v1.0-$(date +%Y%m%d).tar fastdds:gpu

# Create version file
echo "Version: 1.0" > VERSION
echo "Date: $(date)" >> VERSION
echo "CUDA: 11.8" >> VERSION
echo "Fast DDS: 2.14.0" >> VERSION
Additional Resources

Fast DDS Documentation
NVIDIA CUDA Documentation
Docker Documentation

License
MIT License
Contact
For issues or questions, please open an issue on GitHub.

