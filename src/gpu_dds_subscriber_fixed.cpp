#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/topic/Topic.hpp>

#include "GpuData.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace eprosima::fastdds::dds;

class FixedListener : public DataReaderListener
{
private:
    std::atomic<int> matched_{0};
    std::atomic<int> samples_{0};
    
public:
    void on_data_available(DataReader* reader) override
    {
        GpuProcessedData data;
        SampleInfo info;
        
        // Try to read all available samples
        while (reader->take_next_sample(&data, &info) == ReturnCode_t::RETCODE_OK)
        {
            if (info.valid_data)
            {
                samples_++;
                std::cout << "[" << samples_.load() << "] Received: "
                          << "Stream:" << data.stream_id 
                          << " Kernel:" << data.kernel_id 
                          << " Time:" << data.processing_time_ms << "ms"
                          << " DataSize:" << data.data.size() 
                          << " Timestamp:" << data.timestamp_ns << std::endl;
                          
                // Verify data integrity
                if (!data.data.empty())
                {
                    std::cout << "  First data value: " << data.data[0] << std::endl;
                }
            }
        }
    }
    
    void on_subscription_matched(DataReader*, const SubscriptionMatchedStatus& info) override
    {
        matched_ = info.current_count;
        if (info.current_count_change == 1)
        {
            std::cout << ">>> Publisher matched! Total: " << matched_.load() << std::endl;
        }
        else if (info.current_count_change == -1)
        {
            std::cout << ">>> Publisher disconnected. Remaining: " << matched_.load() << std::endl;
        }
    }
    
    void on_requested_incompatible_qos(DataReader*, const RequestedIncompatibleQosStatus& status) override
    {
        std::cout << "!!! ERROR: Incompatible QoS detected !!!" << std::endl;
        std::cout << "    Policy ID: " << status.last_policy_id << std::endl;
        std::cout << "    Total incompatible: " << status.total_count << std::endl;
        
        // Policy IDs: 
        // 2 = DEADLINE, 5 = OWNERSHIP, 6 = LIVELINESS, 11 = RELIABILITY
        // 12 = DESTINATION_ORDER, 13 = HISTORY, 14 = RESOURCE_LIMITS
        // 15 = DURABILITY, 21 = LATENCY_BUDGET
        
        switch(status.last_policy_id)
        {
            case 11:
                std::cout << "    Problem: RELIABILITY mismatch (BEST_EFFORT vs RELIABLE)" << std::endl;
                break;
            case 15:
                std::cout << "    Problem: DURABILITY mismatch" << std::endl;
                break;
            case 13:
                std::cout << "    Problem: HISTORY mismatch" << std::endl;
                break;
            default:
                std::cout << "    Check QoS settings" << std::endl;
        }
    }
    
    void on_sample_rejected(DataReader*, const SampleRejectedStatus& status) override
    {
        std::cout << "!!! Sample rejected! Reason: " << status.last_reason 
                  << ", Total: " << status.total_count << std::endl;
    }
    
    void on_sample_lost(DataReader*, const SampleLostStatus& status) override
    {
        std::cout << "!!! Sample lost! Total: " << status.total_count << std::endl;
    }
    
    int get_samples() const { return samples_.load(); }
    int get_matched() const { return matched_.load(); }
};

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "GPU DDS Subscriber (FIXED VERSION)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create participant
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    pqos.name("GpuSubscriberFixed");
    
    std::cout << "Creating participant on domain 0..." << std::endl;
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
    
    if (!participant)
    {
        std::cerr << "Failed to create participant!" << std::endl;
        return 1;
    }
    
    // Register type
    std::cout << "Registering type..." << std::endl;
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    std::cout << "Type registered: " << type->getName() << std::endl;
    
    // Create topic
    std::cout << "Creating topic 'GpuProcessedDataTopic'..." << std::endl;
    Topic* topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
    
    // Create subscriber
    std::cout << "Creating subscriber..." << std::endl;
    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    
    // Configure DataReader QoS - MATCHING THE PUBLISHER
    DataReaderQos rqos = DATAREADER_QOS_DEFAULT;
    rqos.reliability().kind = RELIABLE_RELIABILITY_QOS;  // CHANGED FROM BEST_EFFORT
    rqos.durability().kind = VOLATILE_DURABILITY_QOS;
    rqos.history().kind = KEEP_LAST_HISTORY_QOS;
    rqos.history().depth = 100;
    
    std::cout << "DataReader QoS Configuration:" << std::endl;
    std::cout << "  Reliability: RELIABLE (matching publisher)" << std::endl;
    std::cout << "  Durability: VOLATILE" << std::endl;
    std::cout << "  History: KEEP_LAST(100)" << std::endl;
    
    // Create DataReader with listener
    FixedListener listener;
    std::cout << "Creating DataReader with listener..." << std::endl;
    DataReader* reader = subscriber->create_datareader(topic, rqos, &listener);
    
    if (!reader)
    {
        std::cerr << "Failed to create DataReader!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Waiting for data ===" << std::endl;
    std::cout << "The subscriber is now listening for messages..." << std::endl;
    
    // Main loop
    for (int i = 0; i < 600; ++i) // 60 seconds
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (i % 50 == 0) // Every 5 seconds
        {
            std::cout << "[STATUS] Time: " << (i/10) << "s"
                     << ", Matched publishers: " << listener.get_matched()
                     << ", Messages received: " << listener.get_samples() << std::endl;
        }
    }
    
    std::cout << "\n=== FINAL STATISTICS ===" << std::endl;
    std::cout << "Total messages received: " << listener.get_samples() << std::endl;
    std::cout << "Final matched count: " << listener.get_matched() << std::endl;
    
    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    subscriber->delete_datareader(reader);
    participant->delete_topic(topic);
    participant->delete_subscriber(subscriber);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
