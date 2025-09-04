#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
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

std::atomic<int> received(0);

class TestListener : public DataReaderListener
{
public:
    void on_data_available(DataReader* reader) override
    {
        GpuProcessedData data;
        SampleInfo info;
        
        while (reader->take_next_sample(&data, &info) == ReturnCode_t::RETCODE_OK)
        {
            if (info.valid_data)
            {
                received++;
                std::cout << "✓ Message " << received.load() << " received: "
                          << "stream=" << data.stream_id 
                          << ", kernel=" << data.kernel_id << std::endl;
            }
        }
    }
};

int main()
{
    std::cout << "=== Simple DDS Test ===" << std::endl;
    
    // Create participant
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, PARTICIPANT_QOS_DEFAULT);
    
    // Register type
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    
    // Create topic
    Topic* topic = participant->create_topic("TestTopic", type->getName(), TOPIC_QOS_DEFAULT);
    
    // Create publisher
    Publisher* publisher = participant->create_publisher(PUBLISHER_QOS_DEFAULT);
    DataWriter* writer = publisher->create_datawriter(topic, DATAWRITER_QOS_DEFAULT);
    
    // Create subscriber
    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    TestListener listener;
    DataReader* reader = subscriber->create_datareader(topic, DATAREADER_QOS_DEFAULT, &listener);
    
    // Wait for discovery
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Send test messages
    std::cout << "Sending 5 test messages..." << std::endl;
    for (int i = 0; i < 5; i++)
    {
        GpuProcessedData sample;
        sample.stream_id = i;
        sample.kernel_id = i * 10;
        sample.data.resize(100, 1.0f);
        sample.processing_time_ms = 1.23f;
        sample.timestamp_ns = 123456789;
        
        writer->write(&sample);
        std::cout << "Sent message " << (i+1) << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Wait for reception
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Messages sent: 5" << std::endl;
    std::cout << "Messages received: " << received.load() << std::endl;
    
    if (received.load() == 5)
    {
        std::cout << "✓✓✓ SUCCESS: All messages received!" << std::endl;
    }
    else
    {
        std::cout << "✗✗✗ FAILURE: Lost " << (5 - received.load()) << " messages" << std::endl;
    }
    
    // Cleanup
    publisher->delete_datawriter(writer);
    subscriber->delete_datareader(reader);
    participant->delete_topic(topic);
    participant->delete_publisher(publisher);
    participant->delete_subscriber(subscriber);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return (received.load() == 5) ? 0 : 1;
}
