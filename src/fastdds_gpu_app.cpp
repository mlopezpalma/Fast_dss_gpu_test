// fastdds_gpu_app.cpp - Fast DDS with GPU/CUDA Integration

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <cstring>
#include <map>

// CUDA interface functions
extern "C" {
    int initCuda();
    float executeVectorAdd(const float* h_a, const float* h_b, float* h_c, int n);
    float executeFftTransform(const float* h_input, float* h_output, int n);
    float executeConvolution(const float* h_input, const float* h_kernel, 
                            float* h_output, int n, int kernel_size);
    float executeMatrixMult(const float* h_A, const float* h_B, float* h_C, int width);
}

const int NUM_STREAMS = 4;
const int NUM_KERNELS = 4;
const int DATA_SIZE = 1024 * 1024; // 1M elements

class FastDdsGpuApp {
private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
    
    // Performance metrics
    struct KernelMetrics {
        std::string name;
        float execution_time_ms;
        float throughput_gbps;
        int data_processed;
    };
    
    std::vector<KernelMetrics> metrics_;
    
public:
    FastDdsGpuApp() 
        : rng_(std::chrono::steady_clock::now().time_since_epoch().count())
        , dist_(0.0f, 1.0f) {
    }
    
    bool initialize() {
        std::cout << "Initializing GPU..." << std::endl;
        
        if (initCuda() != 0) {
            std::cerr << "Failed to initialize CUDA!" << std::endl;
            return false;
        }
        
        std::cout << "GPU initialized successfully!" << std::endl;
        return true;
    }
    
    void generateRandomData(std::vector<float>& data) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dist_(rng_);
        }
    }
    
    void runKernel1_VectorAdd(int stream_id) {
        std::cout << "  Stream " << stream_id << " - Kernel 1: Vector Addition" << std::endl;
        
        std::vector<float> a(DATA_SIZE);
        std::vector<float> b(DATA_SIZE);
        std::vector<float> c(DATA_SIZE);
        
        generateRandomData(a);
        generateRandomData(b);
        
        float time_ms = executeVectorAdd(a.data(), b.data(), c.data(), DATA_SIZE);
        
        float data_gb = (3 * DATA_SIZE * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float throughput = data_gb / (time_ms / 1000.0f);
        
        std::cout << "    Time: " << time_ms << " ms, Throughput: " << throughput << " GB/s" << std::endl;
        
        metrics_.push_back({"Vector Add", time_ms, throughput, DATA_SIZE});
    }
    
    void runKernel2_FFT(int stream_id) {
        std::cout << "  Stream " << stream_id << " - Kernel 2: FFT Transform" << std::endl;
        
        std::vector<float> input(DATA_SIZE);
        std::vector<float> output(DATA_SIZE);
        
        generateRandomData(input);
        
        float time_ms = executeFftTransform(input.data(), output.data(), DATA_SIZE);
        
        float data_gb = (2 * DATA_SIZE * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float throughput = data_gb / (time_ms / 1000.0f);
        
        std::cout << "    Time: " << time_ms << " ms, Throughput: " << throughput << " GB/s" << std::endl;
        
        metrics_.push_back({"FFT Transform", time_ms, throughput, DATA_SIZE});
    }
    
    void runKernel3_Convolution(int stream_id) {
        std::cout << "  Stream " << stream_id << " - Kernel 3: 1D Convolution" << std::endl;
        
        std::vector<float> input(DATA_SIZE);
        std::vector<float> output(DATA_SIZE);
        std::vector<float> kernel = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f}; // 5-tap filter
        
        generateRandomData(input);
        
        float time_ms = executeConvolution(input.data(), kernel.data(), 
                                          output.data(), DATA_SIZE, kernel.size());
        
        float data_gb = (2 * DATA_SIZE * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float throughput = data_gb / (time_ms / 1000.0f);
        
        std::cout << "    Time: " << time_ms << " ms, Throughput: " << throughput << " GB/s" << std::endl;
        
        metrics_.push_back({"1D Convolution", time_ms, throughput, DATA_SIZE});
    }
    
    void runKernel4_MatrixMult(int stream_id) {
        std::cout << "  Stream " << stream_id << " - Kernel 4: Matrix Multiplication" << std::endl;
        
        const int width = 1024; // 1024x1024 matrices
        std::vector<float> A(width * width);
        std::vector<float> B(width * width);
        std::vector<float> C(width * width);
        
        generateRandomData(A);
        generateRandomData(B);
        
        float time_ms = executeMatrixMult(A.data(), B.data(), C.data(), width);
        
        float data_gb = (3 * width * width * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);
        float throughput = data_gb / (time_ms / 1000.0f);
        
        std::cout << "    Time: " << time_ms << " ms, Throughput: " << throughput << " GB/s" << std::endl;
        
        metrics_.push_back({"Matrix Mult", time_ms, throughput, width * width});
    }
    
    void run() {
        std::cout << "\n=== Fast DDS GPU Application ===" << std::endl;
        std::cout << "Running 4 Kernels on 4 Streams" << std::endl;
        std::cout << "Data size: " << DATA_SIZE << " elements\n" << std::endl;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Run 10 iterations
        for (int iter = 0; iter < 10; iter++) {
            std::cout << "\n--- Iteration " << (iter + 1) << " ---" << std::endl;
            
            // Simulate 4 streams
            for (int stream = 0; stream < NUM_STREAMS; stream++) {
                // Execute kernel based on stream ID
                switch (stream % NUM_KERNELS) {
                    case 0:
                        runKernel1_VectorAdd(stream);
                        break;
                    case 1:
                        runKernel2_FFT(stream);
                        break;
                    case 2:
                        runKernel3_Convolution(stream);
                        break;
                    case 3:
                        runKernel4_MatrixMult(stream);
                        break;
                }
                
                // Small delay between streams
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Delay between iterations
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>
                         (total_end - total_start).count();
        
        printSummary(total_time);
    }
    
    void printSummary(long total_time_ms) {
        std::cout << "\n=== PERFORMANCE SUMMARY ===" << std::endl;
        std::cout << "Total execution time: " << total_time_ms << " ms" << std::endl;
        
        // Calculate averages per kernel type
        std::map<std::string, std::vector<float>> kernel_times;
        std::map<std::string, std::vector<float>> kernel_throughputs;
        
        for (const auto& m : metrics_) {
            kernel_times[m.name].push_back(m.execution_time_ms);
            kernel_throughputs[m.name].push_back(m.throughput_gbps);
        }
        
        std::cout << "\nAverage performance per kernel:" << std::endl;
        for (const auto& kt : kernel_times) {
            const std::string& name = kt.first;
            const std::vector<float>& times = kt.second;
            
            float avg_time = 0;
            float avg_throughput = 0;
            
            for (size_t i = 0; i < times.size(); i++) {
                avg_time += times[i];
                avg_throughput += kernel_throughputs[name][i];
            }
            
            avg_time /= times.size();
            avg_throughput /= times.size();
            
            std::cout << "  " << name << ":" << std::endl;
            std::cout << "    Avg time: " << avg_time << " ms" << std::endl;
            std::cout << "    Avg throughput: " << avg_throughput << " GB/s" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Fast DDS with GPU/CUDA Support" << std::endl;
    std::cout << "4 Streams, 4 Kernels" << std::endl;
    std::cout << "========================================" << std::endl;
    
    FastDdsGpuApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application!" << std::endl;
        return 1;
    }
    
    app.run();
    
    std::cout << "\nApplication completed successfully!" << std::endl;
    
    return 0;
}
