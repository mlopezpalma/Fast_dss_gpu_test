// gpu_kernels.cu - CUDA Kernels for Fast DDS
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

#define BLOCK_SIZE 256

// Kernel 1: Vector Addition
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel 2: FFT-like transform
__global__ void fftTransformKernel(const float* input, float* output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = sinf(2.0f * M_PI * val) * cosf(M_PI * val / 2.0f);
    }
}

// Kernel 3: 1D Convolution
__global__ void convolution1DKernel(const float* input, float* output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n && idx > 0 && idx < n-1) {
        output[idx] = 0.25f * input[idx-1] + 0.5f * input[idx] + 0.25f * input[idx+1];
    } else if (idx < n) {
        output[idx] = input[idx];
    }
}

// Kernel 4: Parallel Reduction
__global__ void reductionKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// C++ interface functions
extern "C" {
    void runVectorAdd(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorAddKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
    }
    
    void runFFTTransform(const float* input, float* output, int n, cudaStream_t stream) {
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fftTransformKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(input, output, n);
    }
    
    void runConvolution1D(const float* input, float* output, int n, cudaStream_t stream) {
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        convolution1DKernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(input, output, n);
    }
    
    void runReduction(const float* input, float* output, int n, cudaStream_t stream) {
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reductionKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(input, output, n);
    }
}
