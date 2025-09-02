// gpu_kernels.cu - 4 CUDA Kernels for Fast DDS

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCK_SIZE 256

// Kernel 1: Vector Addition
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel 2: FFT Transform (simplified)
__global__ void fftTransformKernel(const float* input, float* output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float angle = -2.0f * M_PI * idx / n;
        output[idx] = input[idx] * cosf(angle) + input[idx] * sinf(angle);
    }
}

// Kernel 3: 1D Convolution
__global__ void convolution1DKernel(const float* input, const float* kernel, 
                                    float* output, int n, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int k = -half_kernel; k <= half_kernel; k++) {
            int input_idx = idx + k;
            if (input_idx >= 0 && input_idx < n) {
                sum += input[input_idx] * kernel[k + half_kernel];
            }
        }
        output[idx] = sum;
    }
}

// Kernel 4: Matrix Multiplication (square matrices)
__global__ void matrixMultKernel(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// C++ Interface Functions
extern "C" {
    
    // Initialize CUDA
    int initCuda() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
        if (deviceCount == 0) {
            printf("No CUDA devices found!\n");
            return -1;
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        
        return 0;
    }
    
    // Execute Kernel 1: Vector Add
    float executeVectorAdd(const float* h_a, const float* h_b, float* h_c, int n) {
        float *d_a, *d_b, *d_c;
        size_t size = n * sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);
        
        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Copy data to GPU
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
        
        // Execute kernel
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        cudaEventRecord(start);
        vectorAddKernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        
        // Copy result back
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
        
        // Calculate time
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
    
    // Execute Kernel 2: FFT Transform
    float executeFftTransform(const float* h_input, float* h_output, int n) {
        float *d_input, *d_output;
        size_t size = n * sizeof(float);
        
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        cudaEventRecord(start);
        fftTransformKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
    
    // Execute Kernel 3: Convolution
    float executeConvolution(const float* h_input, const float* h_kernel, 
                            float* h_output, int n, int kernel_size) {
        float *d_input, *d_kernel, *d_output;
        
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_kernel, kernel_size * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
        
        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        cudaEventRecord(start);
        convolution1DKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_kernel, d_output, n, kernel_size);
        cudaEventRecord(stop);
        
        cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
    
    // Execute Kernel 4: Matrix Multiplication
    float executeMatrixMult(const float* h_A, const float* h_B, float* h_C, int width) {
        float *d_A, *d_B, *d_C;
        size_t size = width * width * sizeof(float);
        
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
        
        cudaEventRecord(start);
        matrixMultKernel<<<grid, block>>>(d_A, d_B, d_C, width);
        cudaEventRecord(stop);
        
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
}
