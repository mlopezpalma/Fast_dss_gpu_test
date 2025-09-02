#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/topic/Topic.hpp>

#include "GpuData.hpp"

#include <iostream>
#include <thread>
#include <chrono>

using namespace eprosima::fastdds::dds;

class SubListener : public DataReaderListener
{
public:
    int matched_ = 0;
    int samples_ = 0;

    void on_data_available(DataReader* reader) override
    {
        SampleInfo info;
        GpuProcessedData data;
        
        if (reader->take_next_sample(&data, &info) == ReturnCode_t::RETCODE_OK)
        {
            if (info.valid_data)
            {
                samples_++;
                std::cout << "[" << samples_ << "] Stream:" << data.stream_id 
                         << " Kernel:" << data.kernel_id 
                         << " Time:" << data.processing_time_ms << "ms"
                         << " DataSize:" << data.data.size() << std::endl;
            }
        }
    }

    void on_subscription_matched(DataReader*, const SubscriptionMatchedStatus& info) override
    {
        matched_ = info.current_count;
        std::cout << "Matched publishers: " << matched_ << std::endl;
    }
};

int main()
{
    std::cout << "=== GPU DDS Subscriber V2 ===" << std::endl;

    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(0, PARTICIPANT_QOS_DEFAULT);
    
    TypeSupport type(new GpuProcessedDataPubSubType());
    type.register_type(participant);
    
    Topic* topic = participant->create_topic("GpuProcessedDataTopic", type->getName(), TOPIC_QOS_DEFAULT);
    
    Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    
    SubListener listener;
    DataReader* reader = subscriber->create_datareader(topic, DATAREADER_QOS_DEFAULT, &listener);

    std::this_thread::sleep_for(std::chrono::seconds(30));

    std::cout << "Total received: " << listener.samples_ << std::endl;

    return 0;
}
