#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>

#include "GpuData.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;

extern "C" {
    void runVectorAdd(const float* a, const float* b, float* c, int n, cudaStream_t stream);
    void runFFTTransform(const float* input, float* output, int n, cudaStream_t stream);
    void runConvolution1D(const float* input, float* output, int n, cudaStream_t stream);
    void runReduction(const float* input, float* output, int n, cudaStream_t stream);
}

class ParallelBenchmark {
private:
    static const int NUM_STREAMS = 4;
    static const int DATA_SIZE = 1048576;
    static const int ITERATIONS = 50;
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t startEvents[NUM_STREAMS];
    cudaEvent_t stopEvents[NUM_STREAMS];
    cudaEvent_t globalStart, globalStop;
    
    float* d_input_a[NUM_STREAMS];
    float* d_input_b[NUM_STREAMS];
    float* d_output[NUM_STREAMS];
    float* h_input_a[NUM_STREAMS];
    float* h_input_b[NUM_STREAMS];
    float* h_output[NUM_STREAMS];
    
    DomainParticipant* participant;
    Publisher* publisher;
    Topic* topic;
    DataWriter* writer;
    TypeSupport type;
    
public:
    ParallelBenchmark() {
        initCUDA();
        initDDS();
    }
    
    ~ParallelBenchmark() {
        cleanup();
    }
    
    void initCUDA() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "YES" : "NO") << std::endl;
        std::cout << "Async Engine Count: " << prop.asyncEngineCount << std::endl;
        
        size_t size = DATA_SIZE * sizeof(float);
        
        // Create streams and allocate memory
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&startEvents[i]);
            cudaEventCreate(&stopEvents[i]);
            
            // Device memory
            cudaMalloc(&d_input_a[i], size);
            cudaMalloc(&d_input_b[i], size);
            cudaMalloc(&d_output[i], size);
            
            // Pinned host memory for async transfers
            cudaMallocHost(&h_input_a[i], size);
            cudaMallocHost(&h_input_b[i], size);
            cudaMallocHost(&h_output[i], size);
        }
        
        cudaEventCreate(&globalStart);
        cudaEventCreate(&globalStop);
        
        std::cout << "CUDA initialized: " << NUM_STREAMS << " concurrent streams" << std::endl;
    }
    
    void initDDS() {
        DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
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
    
    void runSequentialBenchmark() {
        std::cout << "\n=== SEQUENTIAL EXECUTION ===" << std::endl;
        std::vector<float> times;
        
        for (int iter = 0; iter < ITERATIONS; iter++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int s = 0; s < NUM_STREAMS; s++) {
                // Prepare data
                for (int i = 0; i < DATA_SIZE; i++) {
                    h_input_a[s][i] = static_cast<float>(rand()) / RAND_MAX;
                    h_input_b[s][i] = static_cast<float>(rand()) / RAND_MAX;
                }
                
                // Copy to GPU
                cudaMemcpyAsync(d_input_a[s], h_input_a[s], DATA_SIZE * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_input_b[s], h_input_b[s], DATA_SIZE * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[s]);
                
                // Execute kernel
                runVectorAdd(d_input_a[s], d_input_b[s], d_output[s], DATA_SIZE, streams[s]);
                
                // Copy back
                cudaMemcpyAsync(h_output[s], d_output[s], DATA_SIZE * sizeof(float), 
                               cudaMemcpyDeviceToHost, streams[s]);
                
                // WAIT for this stream to complete before starting next
                cudaStreamSynchronize(streams[s]);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            times.push_back(time_ms);
            
            if (iter % 10 == 0) {
                std::cout << "  Iteration " << iter << ": " << time_ms << " ms" << std::endl;
            }
        }
        
        printStats("Sequential", times);
    }
    
    void runParallelBenchmark() {
        std::cout << "\n=== PARALLEL EXECUTION ===" << std::endl;
        std::vector<float> times;
        
        for (int iter = 0; iter < ITERATIONS; iter++) {
            // Prepare all data first
            for (int s = 0; s < NUM_STREAMS; s++) {
                for (int i = 0; i < DATA_SIZE; i++) {
                    h_input_a[s][i] = static_cast<float>(rand()) / RAND_MAX;
                    h_input_b[s][i] = static_cast<float>(rand()) / RAND_MAX;
                }
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            cudaEventRecord(globalStart);
            
            // Launch ALL streams without waiting
            for (int s = 0; s < NUM_STREAMS; s++) {
                // Copy to GPU
                cudaMemcpyAsync(d_input_a[s], h_input_a[s], DATA_SIZE * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_input_b[s], h_input_b[s], DATA_SIZE * sizeof(float), 
                               cudaMemcpyHostToDevice, streams[s]);
                
                // Execute kernel
                cudaEventRecord(startEvents[s], streams[s]);
                runVectorAdd(d_input_a[s], d_input_b[s], d_output[s], DATA_SIZE, streams[s]);
                cudaEventRecord(stopEvents[s], streams[s]);
                
                // Copy back
                cudaMemcpyAsync(h_output[s], d_output[s], DATA_SIZE * sizeof(float), 
                               cudaMemcpyDeviceToHost, streams[s]);
            }
            
            // Wait for ALL streams to complete
            for (int s = 0; s < NUM_STREAMS; s++) {
                cudaStreamSynchronize(streams[s]);
            }
            
            cudaEventRecord(globalStop);
            cudaEventSynchronize(globalStop);
            
            auto end = std::chrono::high_resolution_clock::now();
            float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            times.push_back(time_ms);
            
            if (iter % 10 == 0) {
                std::cout << "  Iteration " << iter << ": " << time_ms << " ms";
                
                // Show individual stream times
                float gpu_time;
                cudaEventElapsedTime(&gpu_time, globalStart, globalStop);
                std::cout << " (GPU: " << gpu_time << " ms)" << std::endl;
            }
        }
        
        printStats("Parallel", times);
    }
    
    void runOverlappedBenchmark() {
        std::cout << "\n=== OVERLAPPED EXECUTION (Pipeline) ===" << std::endl;
        std::vector<float> times;
        
        for (int iter = 0; iter < ITERATIONS; iter++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Pipeline pattern: each stream does different stage
            for (int batch = 0; batch < 4; batch++) {
                for (int s = 0; s < NUM_STREAMS; s++) {
                    int stage = (batch + s) % 4;
                    
                    switch(stage) {
                        case 0: // Prepare data
                            for (int i = 0; i < DATA_SIZE; i++) {
                                h_input_a[s][i] = static_cast<float>(rand()) / RAND_MAX;
                                h_input_b[s][i] = static_cast<float>(rand()) / RAND_MAX;
                            }
                            break;
                            
                        case 1: // H2D transfer
                            cudaMemcpyAsync(d_input_a[s], h_input_a[s], DATA_SIZE * sizeof(float), 
                                          cudaMemcpyHostToDevice, streams[s]);
                            cudaMemcpyAsync(d_input_b[s], h_input_b[s], DATA_SIZE * sizeof(float), 
                                          cudaMemcpyHostToDevice, streams[s]);
                            break;
                            
                        case 2: // Compute
                            runVectorAdd(d_input_a[s], d_input_b[s], d_output[s], DATA_SIZE, streams[s]);
                            break;
                            
                        case 3: // D2H transfer
                            cudaMemcpyAsync(h_output[s], d_output[s], DATA_SIZE * sizeof(float), 
                                          cudaMemcpyDeviceToHost, streams[s]);
                            break;
                    }
                }
            }
            
            cudaDeviceSynchronize();
            
            auto end = std::chrono::high_resolution_clock::now();
            float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            times.push_back(time_ms);
            
            if (iter % 10 == 0) {
                std::cout << "  Iteration " << iter << ": " << time_ms << " ms" << std::endl;
            }
        }
        
        printStats("Overlapped", times);
    }
    
    void printStats(const std::string& name, const std::vector<float>& times) {
        float sum = 0;
        float min = times[0];
        float max = times[0];
        
        for (float t : times) {
            sum += t;
            if (t < min) min = t;
            if (t > max) max = t;
        }
        
        float avg = sum / times.size();
        
        float variance = 0;
        for (float t : times) {
            variance += (t - avg) * (t - avg);
        }
        float stdev = std::sqrt(variance / times.size());
        
        float data_gb = (NUM_STREAMS * 3 * DATA_SIZE * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        float throughput = data_gb / (avg / 1000.0);
        
        std::cout << "\n" << name << " Results:" << std::endl;
        std::cout << "  Average: " << std::fixed << std::setprecision(3) << avg << " ms" << std::endl;
        std::cout << "  StdDev:  " << stdev << " ms" << std::endl;
        std::cout << "  Min:     " << min << " ms" << std::endl;
        std::cout << "  Max:     " << max << " ms" << std::endl;
        std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
        std::cout << "  Data processed per iteration: " << data_gb << " GB" << std::endl;
    }
    
    void run() {
        std::cout << "\n=== GPU STREAMS BENCHMARK ===" << std::endl;
        std::cout << "Streams: " << NUM_STREAMS << std::endl;
        std::cout << "Data size per stream: " << (DATA_SIZE * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Iterations: " << ITERATIONS << std::endl;
        
        // Warmup
        std::cout << "\nWarming up..." << std::endl;
        for (int i = 0; i < 10; i++) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                runVectorAdd(d_input_a[s], d_input_b[s], d_output[s], DATA_SIZE, streams[s]);
            }
        }
        cudaDeviceSynchronize();
        
        runSequentialBenchmark();
        runParallelBenchmark();
        runOverlappedBenchmark();
        
        std::cout << "\n=== SPEEDUP ANALYSIS ===" << std::endl;
        std::cout << "Compare the average times above to see the benefit of parallel streams" << std::endl;
    }
    
    void cleanup() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaFree(d_input_a[i]);
            cudaFree(d_input_b[i]);
            cudaFree(d_output[i]);
            cudaFreeHost(h_input_a[i]);
            cudaFreeHost(h_input_b[i]);
            cudaFreeHost(h_output[i]);
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(startEvents[i]);
            cudaEventDestroy(stopEvents[i]);
        }
        
        cudaEventDestroy(globalStart);
        cudaEventDestroy(globalStop);
        
        if (writer) publisher->delete_datawriter(writer);
        if (topic) participant->delete_topic(topic);
        if (publisher) participant->delete_publisher(publisher);
        if (participant) DomainParticipantFactory::get_instance()->delete_participant(participant);
    }
};

int main() {
    ParallelBenchmark benchmark;
    benchmark.run();
    return 0;
}
