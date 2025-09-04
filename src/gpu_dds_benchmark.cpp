#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>

#include "GpuData.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <map>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <fstream>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;

// External CUDA functions
extern "C" {
    void runVectorAdd(const float* a, const float* b, float* c, int n, cudaStream_t stream);
    void runFFTTransform(const float* input, float* output, int n, cudaStream_t stream);
    void runConvolution1D(const float* input, float* output, int n, cudaStream_t stream);
    void runReduction(const float* input, float* output, int n, cudaStream_t stream);
}

struct BenchmarkMetrics {
    std::vector<float> gpu_times_ms;
    std::vector<float> cpu_times_ms;
    std::vector<float> dds_send_times_ms;
    std::vector<float> total_times_ms;
    std::vector<float> throughput_gbps;
    
    void add_measurement(float gpu_time, float cpu_time, float dds_time, float total_time, float throughput) {
        gpu_times_ms.push_back(gpu_time);
        cpu_times_ms.push_back(cpu_time);
        dds_send_times_ms.push_back(dds_time);
        total_times_ms.push_back(total_time);
        throughput_gbps.push_back(throughput);
    }
    
    void print_statistics(const std::string& kernel_name) {
        if (gpu_times_ms.empty()) return;
        
        auto compute_stats = [](const std::vector<float>& data) {
            float sum = std::accumulate(data.begin(), data.end(), 0.0f);
            float mean = sum / data.size();
            
            std::vector<float> sorted_data = data;
            std::sort(sorted_data.begin(), sorted_data.end());
            float median = sorted_data[sorted_data.size() / 2];
            float min = sorted_data.front();
            float max = sorted_data.back();
            
            float sq_sum = 0;
            for (float val : data) {
                sq_sum += (val - mean) * (val - mean);
            }
            float stdev = std::sqrt(sq_sum / data.size());
            
            return std::make_tuple(mean, median, min, max, stdev);
        };
        
        auto [gpu_mean, gpu_median, gpu_min, gpu_max, gpu_stdev] = compute_stats(gpu_times_ms);
        auto [cpu_mean, cpu_median, cpu_min, cpu_max, cpu_stdev] = compute_stats(cpu_times_ms);
        auto [dds_mean, dds_median, dds_min, dds_max, dds_stdev] = compute_stats(dds_send_times_ms);
        auto [total_mean, total_median, total_min, total_max, total_stdev] = compute_stats(total_times_ms);
        auto [tp_mean, tp_median, tp_min, tp_max, tp_stdev] = compute_stats(throughput_gbps);
        
        std::cout << "\n=== " << kernel_name << " Statistics ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Samples: " << gpu_times_ms.size() << std::endl;
        std::cout << "\nGPU Kernel Time (ms):" << std::endl;
        std::cout << "  Mean:   " << gpu_mean << " ± " << gpu_stdev << std::endl;
        std::cout << "  Median: " << gpu_median << std::endl;
        std::cout << "  Min:    " << gpu_min << std::endl;
        std::cout << "  Max:    " << gpu_max << std::endl;
        
        std::cout << "\nCPU Time (ms):" << std::endl;
        std::cout << "  Mean:   " << cpu_mean << " ± " << cpu_stdev << std::endl;
        std::cout << "  Median: " << cpu_median << std::endl;
        
        std::cout << "\nDDS Send Time (ms):" << std::endl;
        std::cout << "  Mean:   " << dds_mean << " ± " << dds_stdev << std::endl;
        std::cout << "  Median: " << dds_median << std::endl;
        
        std::cout << "\nTotal Pipeline Time (ms):" << std::endl;
        std::cout << "  Mean:   " << total_mean << " ± " << total_stdev << std::endl;
        std::cout << "  Median: " << total_median << std::endl;
        
        std::cout << "\nThroughput (GB/s):" << std::endl;
        std::cout << "  Mean:   " << tp_mean << " ± " << tp_stdev << std::endl;
        std::cout << "  Median: " << tp_median << std::endl;
        std::cout << "  Max:    " << tp_max << std::endl;
    }
    
    void save_to_csv(const std::string& filename, const std::string& kernel_name) {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) return;
        
        // Write header if file is empty
        file.seekp(0, std::ios::end);
        if (file.tellp() == 0) {
            file << "kernel,gpu_ms,cpu_ms,dds_ms,total_ms,throughput_gbps\n";
        }
        
        for (size_t i = 0; i < gpu_times_ms.size(); i++) {
            file << kernel_name << ","
                 << gpu_times_ms[i] << ","
                 << cpu_times_ms[i] << ","
                 << dds_send_times_ms[i] << ","
                 << total_times_ms[i] << ","
                 << throughput_gbps[i] << "\n";
        }
        file.close();
    }
};

class GpuDdsBenchmark {
private:
    static const int NUM_STREAMS = 4;
    static const int NUM_KERNELS = 4;
    static const int DATA_SIZE = 1048576; // 1M floats = 4MB
    static const int ITERATIONS = 100;
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t startEvents[NUM_STREAMS];
    cudaEvent_t stopEvents[NUM_STREAMS];
    
    float* d_input_a[NUM_STREAMS];
    float* d_input_b[NUM_STREAMS];
    float* d_output[NUM_STREAMS];
    
    DomainParticipant* participant;
    Publisher* publisher;
    Topic* topic;
    DataWriter* writer;
    TypeSupport type;
    
    std::map<std::string, BenchmarkMetrics> metrics;
    
public:
    GpuDdsBenchmark() {
        initCUDA();
        initDDS();
    }
    
    ~GpuDdsBenchmark() {
        cleanup();
    }
    
    void initCUDA() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!\n";
            exit(1);
        }
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "GPU Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        std::cout << "GPU Clock: " << (prop.clockRate / 1000.0) << " MHz" << std::endl;
        
        size_t size = DATA_SIZE * sizeof(float);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&startEvents[i]);
            cudaEventCreate(&stopEvents[i]);
            
            cudaMalloc(&d_input_a[i], size);
            cudaMalloc(&d_input_b[i], size);
            cudaMalloc(&d_output[i], size);
        }
    }
    
    void initDDS() {
        // Configure with UDP only transport
        DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
        pqos.name("BenchmarkPublisher");
        pqos.transport().use_builtin_transports = false;
        
        auto udp_transport = std::make_shared<UDPv4TransportDescriptor>();
        udp_transport->sendBufferSize = 65536;
        udp_transport->receiveBufferSize = 65536;
        pqos.transport().user_transports.push_back(udp_transport);
        
        participant = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
        
        type = TypeSupport(new GpuProcessedDataPubSubType());
        type.register_type(participant);
        
        topic = participant->create_topic("BenchmarkTopic", type->getName(), TOPIC_QOS_DEFAULT);
        publisher = participant->create_publisher(PUBLISHER_QOS_DEFAULT);
        writer = publisher->create_datawriter(topic, DATAWRITER_QOS_DEFAULT);
    }
    
    void runBenchmark() {
        std::cout << "\n=== GPU-DDS Benchmark Starting ===" << std::endl;
        std::cout << "Data size: " << DATA_SIZE << " floats (" << (DATA_SIZE * 4 / (1024.0 * 1024.0)) << " MB)" << std::endl;
        std::cout << "Iterations: " << ITERATIONS << std::endl;
        std::cout << "Streams: " << NUM_STREAMS << std::endl;
        
        std::vector<float> h_input_a(DATA_SIZE);
        std::vector<float> h_input_b(DATA_SIZE);
        std::vector<float> h_output(DATA_SIZE);
        
        const char* kernel_names[] = {"VectorAdd", "FFTTransform", "Convolution1D", "Reduction"};
        
        // Warmup
        std::cout << "\nWarming up..." << std::endl;
        for (int i = 0; i < 10; i++) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                runVectorAdd(d_input_a[s], d_input_b[s], d_output[s], DATA_SIZE, streams[s]);
            }
        }
        cudaDeviceSynchronize();
        
        std::cout << "\nBenchmarking..." << std::endl;
        
        for (int kernel_id = 0; kernel_id < NUM_KERNELS; kernel_id++) {
            std::cout << "\nTesting kernel: " << kernel_names[kernel_id] << std::endl;
            
            for (int iter = 0; iter < ITERATIONS; iter++) {
                if (iter % 20 == 0) {
                    std::cout << "  Progress: " << iter << "/" << ITERATIONS << std::endl;
                }
                
                for (int stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
                    auto total_start = std::chrono::high_resolution_clock::now();
                    
                    // Generate test data
                    auto cpu_start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < DATA_SIZE; i++) {
                        h_input_a[i] = static_cast<float>(rand()) / RAND_MAX;
                        h_input_b[i] = static_cast<float>(rand()) / RAND_MAX;
                    }
                    auto cpu_end = std::chrono::high_resolution_clock::now();
                    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
                    
                    // Copy to GPU and execute kernel
                    cudaMemcpyAsync(d_input_a[stream_id], h_input_a.data(), 
                                   DATA_SIZE * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[stream_id]);
                    cudaMemcpyAsync(d_input_b[stream_id], h_input_b.data(), 
                                   DATA_SIZE * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[stream_id]);
                    
                    cudaEventRecord(startEvents[stream_id], streams[stream_id]);
                    
                    // Execute kernel
                    switch(kernel_id) {
                        case 0:
                            runVectorAdd(d_input_a[stream_id], d_input_b[stream_id], 
                                       d_output[stream_id], DATA_SIZE, streams[stream_id]);
                            break;
                        case 1:
                            runFFTTransform(d_input_a[stream_id], d_output[stream_id], 
                                          DATA_SIZE, streams[stream_id]);
                            break;
                        case 2:
                            runConvolution1D(d_input_a[stream_id], d_output[stream_id], 
                                           DATA_SIZE, streams[stream_id]);
                            break;
                        case 3:
                            runReduction(d_input_a[stream_id], d_output[stream_id], 
                                       DATA_SIZE, streams[stream_id]);
                            break;
                    }
                    
                    cudaEventRecord(stopEvents[stream_id], streams[stream_id]);
                    
                    // Copy back
                    cudaMemcpyAsync(h_output.data(), d_output[stream_id], 
                                   DATA_SIZE * sizeof(float), 
                                   cudaMemcpyDeviceToHost, streams[stream_id]);
                    
                    cudaStreamSynchronize(streams[stream_id]);
                    
                    // Get GPU time
                    float gpu_time;
                    cudaEventElapsedTime(&gpu_time, startEvents[stream_id], stopEvents[stream_id]);
                    
                    // Send via DDS
                    auto dds_start = std::chrono::high_resolution_clock::now();
                    GpuProcessedData sample;
                    sample.stream_id = stream_id;
                    sample.kernel_id = kernel_id;
                    sample.data.assign(h_output.begin(), h_output.begin() + 1024);
                    sample.processing_time_ms = gpu_time;
                    sample.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
                    
                    writer->write(&sample);
                    auto dds_end = std::chrono::high_resolution_clock::now();
                    float dds_time = std::chrono::duration<float, std::milli>(dds_end - dds_start).count();
                    
                    auto total_end = std::chrono::high_resolution_clock::now();
                    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
                    
                    // Calculate throughput
                    float data_gb = (3 * DATA_SIZE * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
                    float throughput = data_gb / (gpu_time / 1000.0);
                    
                    // Store metrics
                    std::string key = std::string(kernel_names[kernel_id]) + "_stream" + std::to_string(stream_id);
                    metrics[key].add_measurement(gpu_time, cpu_time, dds_time, total_time, throughput);
                }
            }
        }
        
        printResults();
    }
    
    void printResults() {
        std::cout << "\n\n=== BENCHMARK RESULTS ===" << std::endl;
        
        for (const auto& [key, metric] : metrics) {
            metric.print_statistics(key);
            metric.save_to_csv("benchmark_results.csv", key);
        }
        
        std::cout << "\nResults saved to benchmark_results.csv" << std::endl;
    }
    
    void cleanup() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaFree(d_input_a[i]);
            cudaFree(d_input_b[i]);
            cudaFree(d_output[i]);
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(startEvents[i]);
            cudaEventDestroy(stopEvents[i]);
        }
        
        if (writer) publisher->delete_datawriter(writer);
        if (topic) participant->delete_topic(topic);
        if (publisher) participant->delete_publisher(publisher);
        if (participant) DomainParticipantFactory::get_instance()->delete_participant(participant);
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "GPU-DDS Performance Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    GpuDdsBenchmark benchmark;
    benchmark.runBenchmark();
    
    return 0;
}
