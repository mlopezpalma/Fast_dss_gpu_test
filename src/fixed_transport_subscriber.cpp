#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>

#include "GpuData.hpp"

#include <iostream>
#include <thread>
#include <chrono>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;

class Listener : public DataReaderListener
{
    int count = 0;
public:
    void on_data_available(DataReader* reader) override
    {
        GpuProcessedData data;
        SampleInfo info;
        
        while (reader->take_next_sample(&data, &info) == ReturnCode_t::RETCODE_OK)
        {
            if (info.valid_data)
            {
                count++;
                std::cout << "[" << count << "] Received: stream=" << data.stream_id 
                          << ", kernel=" << data.kernel_id << std::endl;
            }
        }
    }
    
    void on_subscription_matched(DataReader*, const SubscriptionMatchedStatus& info) override
    {
        std::cout << "Match event: " << info.current_count << " publishers" << std::endl;
    }
};

int main()
{
    std::cout << "=== Subscriber with Fixed Transport ===" << std::endl;
    
    // Configure participant with explicit transport
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    pqos.name("FixedTransportSubscriber");
    
    // Disable shared memory transport
    pqos.transport().use_builtin_transports = false;
    
    // Add only UDPv4 transport
    auto udp_transport = std::make_shared<UDPv4TransportDescriptor>();
    udp_transport->sendBufferSize = 65536;
    udp_transport->receiveBufferSize = 65536;
    pqos.transport().user_transports.push_back(udp_transport);
    
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
    
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    
    Topic* topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
    
    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    
    Listener listener;
    DataReader* reader = subscriber->create_datareader(topic, DATAREADER_QOS_DEFAULT, &listener);
    
    std::cout << "Waiting for messages..." << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(30));
    
    subscriber->delete_datareader(reader);
    participant->delete_topic(topic);
    participant->delete_subscriber(subscriber);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
