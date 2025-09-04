#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.h>

#include "GpuData.hpp"

#include <iostream>
#include <thread>
#include <chrono>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastdds::rtps;

int main()
{
    std::cout << "=== Publisher with Fixed Transport ===" << std::endl;
    
    // Configure participant with explicit transport
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    pqos.name("FixedTransportPublisher");
    
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
    
    Publisher* publisher = participant->create_publisher(PUBLISHER_QOS_DEFAULT);
    DataWriter* writer = publisher->create_datawriter(topic, DATAWRITER_QOS_DEFAULT);
    
    std::cout << "Waiting for subscribers..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    for (int i = 0; i < 10; i++)
    {
        GpuProcessedData sample;
        sample.stream_id = i;
        sample.kernel_id = i * 10;
        sample.data.resize(100, 1.0f);
        sample.processing_time_ms = 1.23f;
        sample.timestamp_ns = 123456789;
        
        writer->write(&sample);
        std::cout << "Sent message " << (i+1) << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    publisher->delete_datawriter(writer);
    participant->delete_topic(topic);
    participant->delete_publisher(publisher);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
