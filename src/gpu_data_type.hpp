// gpu_data_type.hpp - TypeSupport funcional para Fast DDS
#ifndef GPU_DATA_TYPE_HPP
#define GPU_DATA_TYPE_HPP

#include <fastdds/dds/topic/TopicDataType.hpp>
#include <fastrtps/rtps/common/SerializedPayload.h>
#include <vector>
#include <cstring>

using namespace eprosima::fastdds::dds;
using eprosima::fastrtps::rtps::SerializedPayload_t;
using eprosima::fastrtps::rtps::InstanceHandle_t;

class GpuProcessedData {
public:
    int stream_id{0};
    int kernel_id{0};
    std::vector<float> data;
    float processing_time_ms{0.0f};
    long long timestamp_ns{0};
    
    GpuProcessedData() {
        data.resize(1024, 0.0f);
    }
};

class GpuProcessedDataPubSubType : public TopicDataType {
public:
    GpuProcessedDataPubSubType() {
        setName("GpuProcessedData");
        m_typeSize = sizeof(int) * 2 + sizeof(float) * 1025 + sizeof(long long);
        m_isGetKeyDefined = false;
    }
    
    bool serialize(void* data, SerializedPayload_t* payload) override {
        GpuProcessedData* p = static_cast<GpuProcessedData*>(data);
        
        // Calculate needed size
        uint32_t size = sizeof(p->stream_id) + sizeof(p->kernel_id) + 
                       sizeof(uint32_t) + (p->data.size() * sizeof(float)) +
                       sizeof(p->processing_time_ms) + sizeof(p->timestamp_ns);
        
        payload->data = new unsigned char[size];
        payload->length = size;
        
        unsigned char* ptr = payload->data;
        
        // Serialize each field
        memcpy(ptr, &p->stream_id, sizeof(p->stream_id));
        ptr += sizeof(p->stream_id);
        
        memcpy(ptr, &p->kernel_id, sizeof(p->kernel_id));
        ptr += sizeof(p->kernel_id);
        
        uint32_t data_size = p->data.size();
        memcpy(ptr, &data_size, sizeof(data_size));
        ptr += sizeof(data_size);
        
        memcpy(ptr, p->data.data(), data_size * sizeof(float));
        ptr += data_size * sizeof(float);
        
        memcpy(ptr, &p->processing_time_ms, sizeof(p->processing_time_ms));
        ptr += sizeof(p->processing_time_ms);
        
        memcpy(ptr, &p->timestamp_ns, sizeof(p->timestamp_ns));
        
        return true;
    }
    
    bool deserialize(SerializedPayload_t* payload, void* data) override {
        GpuProcessedData* p = static_cast<GpuProcessedData*>(data);
        unsigned char* ptr = payload->data;
        
        memcpy(&p->stream_id, ptr, sizeof(p->stream_id));
        ptr += sizeof(p->stream_id);
        
        memcpy(&p->kernel_id, ptr, sizeof(p->kernel_id));
        ptr += sizeof(p->kernel_id);
        
        uint32_t data_size;
        memcpy(&data_size, ptr, sizeof(data_size));
        ptr += sizeof(data_size);
        
        p->data.resize(data_size);
        memcpy(p->data.data(), ptr, data_size * sizeof(float));
        ptr += data_size * sizeof(float);
        
        memcpy(&p->processing_time_ms, ptr, sizeof(p->processing_time_ms));
        ptr += sizeof(p->processing_time_ms);
        
        memcpy(&p->timestamp_ns, ptr, sizeof(p->timestamp_ns));
        
        return true;
    }
    
    std::function<uint32_t()> getSerializedSizeProvider(void* data) override {
        return [data]() -> uint32_t {
            GpuProcessedData* p = static_cast<GpuProcessedData*>(data);
            return sizeof(p->stream_id) + sizeof(p->kernel_id) + 
                   sizeof(uint32_t) + (p->data.size() * sizeof(float)) +
                   sizeof(p->processing_time_ms) + sizeof(p->timestamp_ns);
        };
    }
    
    void* createData() override {
        return new GpuProcessedData();
    }
    
    void deleteData(void* data) override {
        delete static_cast<GpuProcessedData*>(data);
    }
    
    bool getKey(void* data, InstanceHandle_t* handle, bool force_md5) override {
        return false;
    }
};

#endif
