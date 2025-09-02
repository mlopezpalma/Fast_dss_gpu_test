// fastdds_publisher.cpp - Fast DDS Publisher Example

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;

// Simple data structure (replace with generated from IDL)
struct SimpleGpuData {
    int stream_id;
    int kernel_id;
    std::vector<float> data;
    float processing_time_ms;
    long long timestamp_ns;
    
    SimpleGpuData() : stream_id(0), kernel_id(0), processing_time_ms(0.0f), timestamp_ns(0) {
        data.resize(1024); // Small test size
    }
};

class GpuPublisher {
private:
    DomainParticipant* participant_;
    Publisher* publisher_;
    Topic* topic_;
    DataWriter* writer_;
    
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
    
public:
    GpuPublisher() 
        : participant_(nullptr)
        , publisher_(nullptr)
        , topic_(nullptr)
        , writer_(nullptr)
        , rng_(std::chrono::steady_clock::now().time_since_epoch().count())
        , dist_(0.0f, 1.0f) {
        
        if (!init()) {
            std::cerr << "Failed to initialize publisher" << std::endl;
        }
    }
    
    ~GpuPublisher() {
        cleanup();
    }
    
    bool init() {
        // Create participant
        DomainParticipantQos pqos;
        pqos.name("FastDDS_GPU_Publisher");
        
        participant_ = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
        if (participant_ == nullptr) {
            return false;
        }
        
        // Create publisher
        publisher_ = participant_->create_publisher(PUBLISHER_QOS_DEFAULT);
        if (publisher_ == nullptr) {
            return false;
        }
        
        // Register type (in real app, use generated type)
        // For now, we'll use a built-in type
        topic_ = participant_->create_topic(
            "GpuDataTopic",
            "SimpleGpuData",  // Type name
            TOPIC_QOS_DEFAULT);
            
        if (topic_ == nullptr) {
            return false;
        }
        
        // Create DataWriter
        DataWriterQos wqos;
        wqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
        wqos.history().kind = KEEP_LAST_HISTORY_QOS;
        wqos.history().depth = 10;
        
        writer_ = publisher_->create_datawriter(topic_, wqos);
        if (writer_ == nullptr) {
            return false;
        }
        
        std::cout << "Fast DDS Publisher initialized successfully" << std::endl;
        std::cout << "Domain ID: 0" << std::endl;
        std::cout << "Topic: GpuDataTopic" << std::endl;
        
        return true;
    }
    
    void cleanup() {
        if (writer_ != nullptr) {
            publisher_->delete_datawriter(writer_);
        }
        if (publisher_ != nullptr) {
            participant_->delete_publisher(publisher_);
        }
        if (topic_ != nullptr) {
            participant_->delete_topic(topic_);
        }
        if (participant_ != nullptr) {
            DomainParticipantFactory::get_instance()->delete_participant(participant_);
        }
    }
    
    void generateData(SimpleGpuData& data, int stream_id, int kernel_id) {
        data.stream_id = stream_id;
        data.kernel_id = kernel_id;
        
        // Generate random data
        for (size_t i = 0; i < data.data.size(); i++) {
            data.data[i] = dist_(rng_);
        }
        
        // Simulate processing time
        data.processing_time_ms = 0.5f + dist_(rng_) * 2.0f;
        
        // Timestamp
        data.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    void publish() {
        std::cout << "\n=== Starting Fast DDS Publishing ===" << std::endl;
        
        const int NUM_STREAMS = 4;
        const int NUM_KERNELS = 4;
        const char* kernel_names[] = {
            "Vector Addition",
            "FFT Transform",
            "1D Convolution",
            "Matrix Multiplication"
        };
        
        SimpleGpuData data;
        int iteration = 0;
        
        while (iteration < 10) {  // Run 10 iterations
            std::cout << "\n--- Iteration " << (iteration + 1) << " ---" << std::endl;
            
            for (int kernel_id = 0; kernel_id < NUM_KERNELS; kernel_id++) {
                std::cout << "Processing kernel: " << kernel_names[kernel_id] << std::endl;
                
                for (int stream_id = 0; stream_id < NUM_STREAMS; stream_id++) {
                    // Generate and "process" data
                    generateData(data, stream_id, kernel_id);
                    
                    // In real implementation, write to DDS
                    // writer_->write(&data);
                    
                    std::cout << "  Stream " << stream_id 
                              << ": Generated " << data.data.size() << " floats"
                              << ", Time: " << data.processing_time_ms << " ms" << std::endl;
                }
                
                // Small delay between kernels
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            iteration++;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        std::cout << "\n=== Publishing completed ===" << std::endl;
    }
    
    void run() {
        std::cout << "Fast DDS GPU Publisher starting..." << std::endl;
        std::cout << "Simulating 4 streams with 4 kernels" << std::endl;
        
        // Wait for subscribers
        std::cout << "Waiting 2 seconds for subscribers..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        publish();
    }
};

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Fast DDS GPU Publisher (Test Version)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        GpuPublisher publisher;
        publisher.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
