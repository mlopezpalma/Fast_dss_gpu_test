#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/topic/Topic.hpp>

#include "GpuData.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdlib>

using namespace eprosima::fastdds::dds;

int main()
{
    std::cout << "=== GPU DDS Publisher (RELIABLE QoS) ===" << std::endl;
    
    // Create participant
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, PARTICIPANT_QOS_DEFAULT);
    
    // Register type
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    std::cout << "Type registered: " << type->getName() << std::endl;
    
    // Create topic
    Topic* topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
    
    // Create publisher
    Publisher* publisher = participant->create_publisher(PUBLISHER_QOS_DEFAULT);
    
    // Create DataWriter with EXPLICIT RELIABLE QoS
    DataWriterQos wqos = DATAWRITER_QOS_DEFAULT;
    wqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
    wqos.durability().kind = VOLATILE_DURABILITY_QOS;
    wqos.history().kind = KEEP_LAST_HISTORY_QOS;
    wqos.history().depth = 100;
    
    std::cout << "DataWriter QoS:" << std::endl;
    std::cout << "  Reliability: RELIABLE" << std::endl;
    std::cout << "  Durability: VOLATILE" << std::endl;
    std::cout << "  History: KEEP_LAST(100)" << std::endl;
    
    DataWriter* writer = publisher->create_datawriter(topic, wqos);
    
    if (!writer)
    {
        std::cerr << "Failed to create DataWriter!" << std::endl;
        return 1;
    }
    
    std::cout << "Waiting 2 seconds for subscribers..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "\n=== Starting to send messages ===" << std::endl;
    
    // Send test messages
    for (int iter = 0; iter < 5; iter++)
    {
        std::cout << "Iteration " << (iter + 1) << std::endl;
        
        for (int kernel = 0; kernel < 4; kernel++)
        {
            for (int stream = 0; stream < 4; stream++)
            {
                GpuProcessedData sample;
                sample.stream_id = stream;
                sample.kernel_id = kernel;
                sample.data.resize(1024);
                for (size_t i = 0; i < sample.data.size(); i++)
                {
                    sample.data[i] = static_cast<float>(rand()) / RAND_MAX;
                }
                sample.processing_time_ms = 1.23f + stream + kernel;
                sample.timestamp_ns = std::chrono::steady_clock::now().time_since_epoch().count();
                
                ReturnCode_t ret = writer->write(&sample);
                if (ret == ReturnCode_t::RETCODE_OK)
                {
                    std::cout << "  Stream " << stream << ", Kernel " << kernel << " [SENT]" << std::endl;
                }
                else
                {
                    std::cout << "  Stream " << stream << ", Kernel " << kernel << " [FAILED: " << ret() << "]" << std::endl;
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\n=== Complete ===" << std::endl;
    std::cout << "Total messages sent: " << (5 * 4 * 4) << std::endl;
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Cleanup
    publisher->delete_datawriter(writer);
    participant->delete_topic(topic);
    participant->delete_publisher(publisher);
    DomainParticipantFactory::get_instance()->delete_participant(participant);
    
    return 0;
}
