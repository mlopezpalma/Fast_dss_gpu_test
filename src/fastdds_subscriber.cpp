// fastdds_subscriber.cpp - Fast DDS Subscriber Example (FIXED)

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/topic/Topic.hpp>

#include <iostream>
#include <thread>
#include <chrono>

using namespace eprosima::fastdds::dds;

// Changed name to avoid conflict with Fast DDS internal SubscriberListener
class GpuDataReaderListener : public DataReaderListener {
public:
    void on_data_available(DataReader* reader) override {
        std::cout << "Data received!" << std::endl;
        // In real implementation, read the data here
    }
    
    void on_subscription_matched(
        DataReader* reader,
        const SubscriptionMatchedStatus& info) override {
        
        if (info.current_count_change > 0) {
            std::cout << "Publisher matched!" << std::endl;
        } else if (info.current_count_change < 0) {
            std::cout << "Publisher unmatched!" << std::endl;
        }
    }
};

class GpuSubscriber {
private:
    DomainParticipant* participant_;
    Subscriber* subscriber_;
    Topic* topic_;
    DataReader* reader_;
    GpuDataReaderListener listener_;  // Using renamed class
    
public:
    GpuSubscriber() 
        : participant_(nullptr)
        , subscriber_(nullptr)
        , topic_(nullptr)
        , reader_(nullptr) {
        
        if (!init()) {
            std::cerr << "Failed to initialize subscriber" << std::endl;
        }
    }
    
    ~GpuSubscriber() {
        cleanup();
    }
    
    bool init() {
        // Create participant
        DomainParticipantQos pqos;
        pqos.name("FastDDS_GPU_Subscriber");
        
        participant_ = DomainParticipantFactory::get_instance()->create_participant(0, pqos);
        if (participant_ == nullptr) {
            return false;
        }
        
        // Create subscriber
        subscriber_ = participant_->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
        if (subscriber_ == nullptr) {
            return false;
        }
        
        // Create topic
        topic_ = participant_->create_topic(
            "GpuDataTopic",
            "SimpleGpuData",
            TOPIC_QOS_DEFAULT);
            
        if (topic_ == nullptr) {
            return false;
        }
        
        // Create DataReader
        DataReaderQos rqos;
        rqos.reliability().kind = RELIABLE_RELIABILITY_QOS;
        rqos.history().kind = KEEP_LAST_HISTORY_QOS;
        rqos.history().depth = 10;
        
        reader_ = subscriber_->create_datareader(topic_, rqos, &listener_);
        if (reader_ == nullptr) {
            return false;
        }
        
        std::cout << "Fast DDS Subscriber initialized successfully" << std::endl;
        std::cout << "Domain ID: 0" << std::endl;
        std::cout << "Topic: GpuDataTopic" << std::endl;
        
        return true;
    }
    
    void cleanup() {
        if (reader_ != nullptr) {
            subscriber_->delete_datareader(reader_);
        }
        if (subscriber_ != nullptr) {
            participant_->delete_subscriber(subscriber_);
        }
        if (topic_ != nullptr) {
            participant_->delete_topic(topic_);
        }
        if (participant_ != nullptr) {
            DomainParticipantFactory::get_instance()->delete_participant(participant_);
        }
    }
    
    void run() {
        std::cout << "Fast DDS GPU Subscriber running..." << std::endl;
        std::cout << "Waiting for publishers..." << std::endl;
        
        // Keep running for 30 seconds for testing
        for (int i = 0; i < 30; i++) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "." << std::flush;
        }
        std::cout << std::endl;
        
        std::cout << "Subscriber stopping after 30 seconds..." << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Fast DDS GPU Subscriber (Test Version)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        GpuSubscriber subscriber;
        subscriber.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
