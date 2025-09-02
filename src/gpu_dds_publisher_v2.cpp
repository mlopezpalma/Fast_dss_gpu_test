#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/topic/Topic.hpp>

#include "GpuData.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <chrono>
#include <thread>

using namespace eprosima::fastdds::dds;

extern "C" {
    void runVectorAdd(const float* a, const float* b, float* c, int n, cudaStream_t stream);
    void runFFTTransform(const float* input, float* output, int n, cudaStream_t stream);
    void runConvolution1D(const float* input, float* output, int n, cudaStream_t stream);
    void runReduction(const float* input, float* output, int n, cudaStream_t stream);
}

class GpuDdsPublisher {
private:
    static const int NUM_STREAMS = 4;
    static const int NUM_KERNELS = 4;
    static const int DATA_SIZE = 1048576;
    
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
    
public:
    GpuDdsPublisher() {
        initCUDA();
        initDDS();
    }
    
    ~GpuDdsPublisher() {
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
        
        size_t size = DATA_SIZE * sizeof(float);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&startEvents[i]);
            cudaEventCreate(&stopEvents[i]);
            
            cudaMalloc(&d_input_a[i], size);
            cudaMalloc(&d_input_b[i], size);
            cudaMalloc(&d_output[i], size);
        }
        
        std::cout << "CUDA initialized with " << NUM_STREAMS << " streams\n";
    }
    
    void initDDS() {
        participant = DomainParticipantFactory::get_instance()->create_participant(0, PARTICIPANT_QOS_DEFAULT);
        
        type = TypeSupport(new GpuProcessedDataPubSubType());
        type.register_type(participant);
        
        topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
        
        publisher = participant->create_publisher(PUBLISHER_QOS_DEFAULT);
        
        writer = publisher->create_datawriter(topic, DATAWRITER_QOS_DEFAULT);
        
        std::cout << "DDS initialized\n";
    }
    
    void run() {
        std::cout << "\n=== Starting GPU Processing ===\n";
        
        std::vector<float> h_input_a(DATA_SIZE);
        std::vector<float> h_input_b(DATA_SIZE);
        std::vector<float> h_output(DATA_SIZE);
        
        for (int iter = 0; iter < 10; iter++) {
            std::cout << "Iteration " << iter + 1 << std::endl;
            
            for (int kernel_id = 0; kernel_id < NUM_KERNELS; kernel_id++) {
                for (int stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
                    
                    for (int i = 0; i < DATA_SIZE; i++) {
                        h_input_a[i] = static_cast<float>(rand()) / RAND_MAX;
                        h_input_b[i] = static_cast<float>(rand()) / RAND_MAX;
                    }
                    
                    cudaMemcpyAsync(d_input_a[stream_id], h_input_a.data(), 
                                   DATA_SIZE * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[stream_id]);
                    cudaMemcpyAsync(d_input_b[stream_id], h_input_b.data(), 
                                   DATA_SIZE * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[stream_id]);
                    
                    cudaEventRecord(startEvents[stream_id], streams[stream_id]);
                    
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
                    
                    cudaMemcpyAsync(h_output.data(), d_output[stream_id], 
                                   DATA_SIZE * sizeof(float), 
                                   cudaMemcpyDeviceToHost, streams[stream_id]);
                    
                    cudaStreamSynchronize(streams[stream_id]);
                    
                    float gpu_time;
                    cudaEventElapsedTime(&gpu_time, startEvents[stream_id], stopEvents[stream_id]);
                    
                    // Use GpuProcessedData from GpuData.hpp
                    GpuProcessedData sample;
                    sample.stream_id = stream_id;
                    sample.kernel_id = kernel_id;
                    sample.data.assign(h_output.begin(), h_output.begin() + 1024);
                    sample.processing_time_ms = gpu_time;
                    sample.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
                    
                    writer->write(&sample);
                    
                    std::cout << "  Stream " << stream_id 
                              << ", Kernel " << kernel_id 
                              << ": " << gpu_time << " ms [SENT]" << std::endl;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n=== Complete ===\n";
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
    std::cout << "=== GPU DDS Publisher V2 ===" << std::endl;
    
    GpuDdsPublisher publisher;
    publisher.run();
    
    return 0;
}
