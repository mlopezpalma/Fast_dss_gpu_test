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
#include <iomanip>

using namespace eprosima::fastdds::dds;

class DebugListener : public DataReaderListener
{
public:
    int matched_ = 0;
    int samples_ = 0;
    int on_data_calls_ = 0;
    
    void on_data_available(DataReader* reader) override
    {
        on_data_calls_++;
        std::cout << "[DEBUG] on_data_available called (#" << on_data_calls_ << ")" << std::endl;
        
        GpuProcessedData data;
        SampleInfo info;
        
        ReturnCode_t ret = reader->take_next_sample(&data, &info);
        std::cout << "[DEBUG] take_next_sample returned: " << ret() << std::endl;
        
        if (ret == ReturnCode_t::RETCODE_OK)
        {
            std::cout << "[DEBUG] Sample info - valid: " << info.valid_data 
                     << ", instance_state: " << info.instance_state
                     << ", sample_state: " << info.sample_state << std::endl;
                     
            if (info.valid_data)
            {
                samples_++;
                std::cout << "[RECEIVED #" << samples_ << "] "
                         << "Stream:" << data.stream_id 
                         << " Kernel:" << data.kernel_id 
                         << " Time:" << data.processing_time_ms << "ms"
                         << " DataSize:" << data.data.size() << std::endl;
            }
            else
            {
                std::cout << "[DEBUG] Data not valid!" << std::endl;
            }
        }
        else if (ret == ReturnCode_t::RETCODE_NO_DATA)
        {
            std::cout << "[DEBUG] No data available" << std::endl;
        }
        else
        {
            std::cout << "[DEBUG] Error taking sample: " << ret() << std::endl;
        }
    }
    
    void on_subscription_matched(DataReader* reader, const SubscriptionMatchedStatus& info) override
    {
        matched_ = info.current_count;
        std::cout << "[MATCH EVENT] Change: " << info.current_count_change 
                 << ", Total matched: " << info.current_count
                 << ", Total change: " << info.total_count_change << std::endl;
                 
        if (info.current_count_change == 1)
        {
            std::cout << "[DEBUG] New publisher detected and matched" << std::endl;
        }
        else if (info.current_count_change == -1)
        {
            std::cout << "[DEBUG] Publisher disconnected" << std::endl;
        }
    }
    
    void on_sample_rejected(DataReader* reader, const SampleRejectedStatus& status) override
    {
        std::cout << "[SAMPLE REJECTED] Reason: " << status.last_reason 
                 << ", Total: " << status.total_count << std::endl;
    }
    
    void on_liveliness_changed(DataReader* reader, const LivelinessChangedStatus& status) override
    {
        std::cout << "[LIVELINESS] Alive: " << status.alive_count 
                 << ", Not alive: " << status.not_alive_count << std::endl;
    }
    
    void on_sample_lost(DataReader* reader, const SampleLostStatus& status) override
    {
        std::cout << "[SAMPLE LOST] Total: " << status.total_count << std::endl;
    }
};

int main()
{
    std::cout << "=== GPU DDS Subscriber DEBUG VERSION ===" << std::endl;
    std::cout << "Starting at: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
    
    // Create participant with debug name
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    pqos.name("DebugSubscriber");
    
    std::cout << "[DEBUG] Creating participant on domain 0..." << std::endl;
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
    
    if (!participant)
    {
        std::cerr << "[ERROR] Failed to create participant!" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] Participant created successfully" << std::endl;
    
    // Register type
    std::cout << "[DEBUG] Registering type..." << std::endl;
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    std::cout << "[DEBUG] Type registered: " << type->getName() << std::endl;
    
    // Create topic
    std::cout << "[DEBUG] Creating topic 'GpuProcessedDataTopic'..." << std::endl;
    Topic* topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
    if (!topic)
    {
        std::cerr << "[ERROR] Failed to create topic!" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] Topic created" << std::endl;
    
    // Create subscriber
    std::cout << "[DEBUG] Creating subscriber..." << std::endl;
    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    if (!subscriber)
    {
        std::cerr << "[ERROR] Failed to create subscriber!" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] Subscriber created" << std::endl;
    
    // Configure DataReader QoS
    DataReaderQos rqos = DATAREADER_QOS_DEFAULT;
    rqos.reliability().kind = BEST_EFFORT_RELIABILITY_QOS;  // Try BEST_EFFORT
    rqos.durability().kind = VOLATILE_DURABILITY_QOS;
    rqos.history().kind = KEEP_LAST_HISTORY_QOS;
    rqos.history().depth = 100;
    
    std::cout << "[DEBUG] DataReader QoS:" << std::endl;
    std::cout << "  Reliability: BEST_EFFORT" << std::endl;
    std::cout << "  Durability: VOLATILE" << std::endl;
    std::cout << "  History: KEEP_LAST(100)" << std::endl;
    
    // Create DataReader
    DebugListener listener;
    std::cout << "[DEBUG] Creating DataReader..." << std::endl;
    DataReader* reader = subscriber->create_datareader(topic, rqos, &listener);
    if (!reader)
    {
        std::cerr << "[ERROR] Failed to create DataReader!" << std::endl;
        return 1;
    }
    std::cout << "[DEBUG] DataReader created" << std::endl;
    
    // Main loop with status
    std::cout << "\n[DEBUG] Entering main loop - waiting for data..." << std::endl;
    for (int i = 0; i < 300; ++i) // 30 seconds
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (i % 50 == 0) // Every 5 seconds
        {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::cout << "[STATUS " << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] "
                     << "Matched: " << listener.matched_ 
                     << ", Received: " << listener.samples_
                     << ", on_data_calls: " << listener.on_data_calls_ << std::endl;
        }
    }
    
    std::cout << "\n=== FINAL STATISTICS ===" << std::endl;
    std::cout << "Total on_data_available calls: " << listener.on_data_calls_ << std::endl;
    std::cout << "Total samples received: " << listener.samples_ << std::endl;
    std::cout << "Final matched count: " << listener.matched_ << std::endl;
    
    // Cleanup
    std::cout << "[DEBUG] Cleaning up..." << std::endl;
    subscriber->delete_datareader(reader);
    participant->delete_topic(topic);
    participant->delete_subscriber(subscriber);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
