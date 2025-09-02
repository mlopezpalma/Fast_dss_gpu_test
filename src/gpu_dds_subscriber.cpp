// gpu_dds_subscriber.cpp - Fast DDS Subscriber corregido
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include "gpu_data_type.hpp"

#include <iostream>
#include <chrono>
#include <thread>

using namespace eprosima::fastdds::dds;

class GpuDataReaderListener : public DataReaderListener {
private:
    int samples_received_{0};
    
public:
    void on_data_available(DataReader* reader) override {
        GpuProcessedData data;
        SampleInfo info;
        
        if (reader->take_next_sample(&data, &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                samples_received_++;
                std::cout << "[" << samples_received_ << "] Received - "
                          << "Stream: " << data.stream_id 
                          << ", Kernel: " << data.kernel_id
                          << ", GPU time: " << data.processing_time_ms << " ms"
                          << ", Data[0]: " << (data.data.empty() ? 0.0f : data.data[0])
                          << std::endl;
            }
        }
    }
    
    void on_subscription_matched(DataReader*, const SubscriptionMatchedStatus& info) override {
        if (info.current_count_change == 1) {
            std::cout << ">>> Matched with publisher <<<" << std::endl;
        } else if (info.current_count_change == -1) {
            std::cout << ">>> Unmatched from publisher <<<" << std::endl;
        }
    }
    
    int get_samples() const { return samples_received_; }
};

int main() {
    std::cout << "========================================\n";
    std::cout << "GPU-DDS Subscriber (Fixed)\n";
    std::cout << "========================================\n";
    
    // Create participant
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, PARTICIPANT_QOS_DEFAULT);
    if (!participant) {
        std::cerr << "Failed to create participant\n";
        return 1;
    }
    
    // Register type
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    std::cout << "Type registered: " << type->getName() << std::endl;
    
    // Create subscriber
    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    
    // Create topic
    Topic* topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
    
    // Create DataReader with listener
    GpuDataReaderListener listener;
    DataReader* reader = subscriber->create_datareader(topic, DATAREADER_QOS_DEFAULT, &listener);
    
    std::cout << "Waiting for data...\n" << std::endl;
    
    // Run for 30 seconds
    for (int i = 0; i < 300; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (i % 50 == 0) {
            std::cout << "Active... (received: " << listener.get_samples() << " messages)" << std::endl;
        }
    }
    
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total messages received: " << listener.get_samples() << std::endl;
    
    // Cleanup
    subscriber->delete_datareader(reader);
    participant->delete_topic(topic);
    participant->delete_subscriber(subscriber);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
