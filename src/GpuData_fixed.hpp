#ifndef _GPUDATA_FIXED_HPP_
#define _GPUDATA_FIXED_HPP_

#include <vector>
#include <fastdds/dds/topic/TopicDataType.hpp>
#include <fastrtps/rtps/common/SerializedPayload.h>
#include <cstring>

class GpuProcessedData
{
public:
    int32_t stream_id{0};
    int32_t kernel_id{0};
    std::vector<float> data;
    float processing_time_ms{0.0f};
    int64_t timestamp_ns{0};

    GpuProcessedData()
    {
        data.reserve(1024);
    }
};

class GpuProcessedDataPubSubType : public eprosima::fastdds::dds::TopicDataType
{
public:
    typedef GpuProcessedData type;

    GpuProcessedDataPubSubType()
    {
        setName("GpuProcessedData");
        auto type_size = sizeof(int32_t) * 2 + sizeof(uint32_t) + 1024 * sizeof(float) + sizeof(float) + sizeof(int64_t);
        m_typeSize = static_cast<uint32_t>(type_size + 4);
        m_isGetKeyDefined = false;
    }

    ~GpuProcessedDataPubSubType() override = default;

    bool serialize(void* data, eprosima::fastrtps::rtps::SerializedPayload_t* payload) override
    {
        GpuProcessedData* p_type = static_cast<GpuProcessedData*>(data);

        uint32_t size = sizeof(p_type->stream_id) + sizeof(p_type->kernel_id) + sizeof(uint32_t) + 
                       p_type->data.size() * sizeof(float) + sizeof(p_type->processing_time_ms) + sizeof(p_type->timestamp_ns);

        if (payload->max_size < size)
        {
            payload->reserve(size);
        }

        payload->length = size;
        
        unsigned char* dst = static_cast<unsigned char*>(payload->data);
        size_t offset = 0;

        memcpy(dst + offset, &p_type->stream_id, sizeof(p_type->stream_id));
        offset += sizeof(p_type->stream_id);

        memcpy(dst + offset, &p_type->kernel_id, sizeof(p_type->kernel_id));
        offset += sizeof(p_type->kernel_id);

        uint32_t data_size = static_cast<uint32_t>(p_type->data.size());
        memcpy(dst + offset, &data_size, sizeof(data_size));
        offset += sizeof(data_size);

        if (data_size > 0)
        {
            memcpy(dst + offset, p_type->data.data(), data_size * sizeof(float));
            offset += data_size * sizeof(float);
        }

        memcpy(dst + offset, &p_type->processing_time_ms, sizeof(p_type->processing_time_ms));
        offset += sizeof(p_type->processing_time_ms);

        memcpy(dst + offset, &p_type->timestamp_ns, sizeof(p_type->timestamp_ns));

        return true;
    }

    bool deserialize(eprosima::fastrtps::rtps::SerializedPayload_t* payload, void* data) override
    {
        GpuProcessedData* p_type = static_cast<GpuProcessedData*>(data);
        
        unsigned char* src = static_cast<unsigned char*>(payload->data);
        size_t offset = 0;

        if (payload->length < (sizeof(int32_t) * 2 + sizeof(uint32_t) + sizeof(float) + sizeof(int64_t)))
        {
            return false;
        }

        memcpy(&p_type->stream_id, src + offset, sizeof(p_type->stream_id));
        offset += sizeof(p_type->stream_id);

        memcpy(&p_type->kernel_id, src + offset, sizeof(p_type->kernel_id));
        offset += sizeof(p_type->kernel_id);

        uint32_t data_size = 0;
        memcpy(&data_size, src + offset, sizeof(data_size));
        offset += sizeof(data_size);

        if (data_size > 1024 * 1024)
        {
            return false;
        }

        p_type->data.resize(data_size);
        if (data_size > 0)
        {
            memcpy(p_type->data.data(), src + offset, data_size * sizeof(float));
            offset += data_size * sizeof(float);
        }

        memcpy(&p_type->processing_time_ms, src + offset, sizeof(p_type->processing_time_ms));
        offset += sizeof(p_type->processing_time_ms);

        memcpy(&p_type->timestamp_ns, src + offset, sizeof(p_type->timestamp_ns));

        return true;
    }

    std::function<uint32_t()> getSerializedSizeProvider(void* data) override
    {
        return [data]() -> uint32_t
        {
            GpuProcessedData* p = static_cast<GpuProcessedData*>(data);
            return sizeof(p->stream_id) + sizeof(p->kernel_id) + sizeof(uint32_t) + 
                   p->data.size() * sizeof(float) + sizeof(p->processing_time_ms) + sizeof(p->timestamp_ns);
        };
    }

    void* createData() override
    {
        return reinterpret_cast<void*>(new GpuProcessedData());
    }

    void deleteData(void* data) override
    {
        delete(reinterpret_cast<GpuProcessedData*>(data));
    }

    bool getKey(void* data, eprosima::fastrtps::rtps::InstanceHandle_t* handle, bool force_md5 = false) override
    {
        return false;
    }
};

#endif
