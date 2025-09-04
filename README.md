# Fast DDS with GPU/CUDA Integration

High-performance data distribution system using eProsima Fast DDS with NVIDIA CUDA GPU acceleration.

## 🚀 Features

- **4 Concurrent CUDA Streams** for parallel GPU processing
- **Multiple GPU Kernels**: Vector Add, FFT, Convolution, Reduction
- **Fast DDS Integration** with reliable QoS
- **Docker Containerization** with full GPU support
- **Benchmark Suite** for performance analysis

## 📋 Prerequisites

- NVIDIA GPU (Compute Capability 7.0+)
- Docker Engine 20.10+
- NVIDIA Container Toolkit
- Ubuntu 20.04/22.04 or compatible Linux
- 8GB+ RAM

## 🛠️ Installation

### Step 1: Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
Step 2: Clone Repository
bashgit clone https://github.com/mlopezpalma/Fast_dss_gpu_test.git
cd fastdds-gpu-project
Step 3: Build Docker Image
bashdocker build -t fastdds:fixed .
Step 4: Verify Installation
bash# Test GPU access
docker run --rm --gpus all fastdds:fixed nvidia-smi

# List built executables
docker run --rm fastdds:fixed ls -la /opt/fastdds-gpu/build/
💻 Usage
Run Publisher-Subscriber Test
Terminal 1 - Start Subscriber:
bashdocker run --rm --network host fastdds:fixed \
    /opt/fastdds-gpu/build/gpu_dds_subscriber_fixed
Terminal 2 - Start Publisher:
bashdocker run --rm --gpus all --network host fastdds:fixed \
    /opt/fastdds-gpu/build/gpu_dds_publisher_v2
Run Performance Benchmark
bashdocker run --rm --gpus all --network host fastdds:fixed \
    /opt/fastdds-gpu/build/gpu_parallel_benchmark
Simple Connection Test
bashdocker run --rm --network host fastdds:fixed \
    /opt/fastdds-gpu/build/simple_test
📊 Performance Results
GPU: NVIDIA GeForce RTX 4060
Execution ModeAverage TimeThroughputSpeedupSequential36.92 ms1.27 GB/s1.0xParallel2.95 ms15.91 GB/s12.5xPipelined34.24 ms1.37 GB/s1.08x
Kernel Performance

Vector Add: 0.025-0.045 ms
FFT Transform: 0.065-0.075 ms
Convolution: 0.032-0.044 ms
Reduction: 0.040-0.050 ms

🗂️ Project Structure
fastdds-gpu-project/
├── Dockerfile                 # Main Docker configuration
├── CMakeLists.txt            # Build configuration
├── src/
│   ├── gpu_dds_publisher_v2.cpp
│   ├── gpu_dds_subscriber_fixed.cpp
│   ├── gpu_kernels.cu       # CUDA kernels
│   ├── GpuData.hpp          # Data structures
│   └── simple_test.cpp      # Connection test
├── benchmark_results/        # Performance results
└── README.md                # This file
🔧 Available Executables
ExecutableDescriptiongpu_dds_publisher_v2GPU-accelerated publishergpu_dds_subscriber_fixedOptimized subscriber with correct QoSgpu_parallel_benchmarkParallel streams benchmarksimple_testBasic DDS connectivity testgpu_dds_publisher_reliablePublisher with reliable QoS
📦 Docker Management
Save Docker Image
bashdocker save -o fastdds_backup.tar fastdds:fixed
gzip fastdds_backup.tar
Load Docker Image
bashgunzip fastdds_backup.tar.gz
docker load -i fastdds_backup.tar
Clean Up
bash# Remove unused images
docker image prune

# Remove build cache
docker builder prune

# Full cleanup (careful!)
docker system prune -a
🐛 Troubleshooting
GPU Not Detected
bash# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# If fails, reinstall toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
No Messages Received
bash# Check network configuration
docker network ls

# Use host network mode
docker run --network host ...
Permission Denied
bash# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
📈 Monitoring
GPU Usage
bash# Real-time GPU monitoring
nvidia-smi -l 1
Docker Stats
bash# Container resource usage
docker stats
🤝 Contributing

Fork the repository
Create your feature branch (git checkout -b feature/amazing)
Commit your changes (git commit -m 'Add feature')
Push to the branch (git push origin feature/amazing)
Open a Pull Request

📄 License
MIT License - see LICENSE file for details
📞 Contact

Repository: https://github.com/mlopezpalma/Fast_dss_gpu_test
Issues: Please open an issue on GitHub

🙏 Acknowledgments

eProsima Fast DDS team
NVIDIA CUDA team
Docker community


Last updated: September 2025
