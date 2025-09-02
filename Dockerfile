FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    libasio-dev \
    libtinyxml2-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Install foonathan memory
RUN git clone https://github.com/foonathan/memory.git && \
    cd memory && \
    git checkout v0.7-3 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf memory

# Install Fast-CDR
RUN git clone https://github.com/eProsima/Fast-CDR.git && \
    cd Fast-CDR && \
    git checkout v2.1.3 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf Fast-CDR

# Install Fast-DDS
RUN git clone https://github.com/eProsima/Fast-DDS.git && \
    cd Fast-DDS && \
    git checkout v2.14.0 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && rm -rf Fast-DDS

# Set environment
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

# Create project directory
WORKDIR /opt/fastdds-gpu

# Copy all source files
COPY . .

# Build the project with CUDA support
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Verify executables
RUN ls -la /opt/fastdds-gpu/build/

WORKDIR /opt/fastdds-gpu

CMD ["bash"]
