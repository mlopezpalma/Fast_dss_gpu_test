#ifndef _GPUDATA_HPP_
#define _GPUDATA_HPP_

#include <vector>
#include <fastdds/dds/topic/TopicDataType.hpp>
#include <fastrtps/rtps/common/SerializedPayload.h>
#include <fastrtps/utils/md5.h>

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
        auto type_size = sizeof(GpuProcessedData);
        type_size += 1024 * sizeof(float);
        m_typeSize = static_cast<uint32_t>(type_size);
        m_isGetKeyDefined = false;
    }

    ~GpuProcessedDataPubSubType() override = default;

    bool serialize(
            void* data,
            eprosima::fastrtps::rtps::SerializedPayload_t* payload) override
    {
        GpuProcessedData* p_type = static_cast<GpuProcessedData*>(data);

        uint32_t size = 0;
        size += 4; // stream_id
        size += 4; // kernel_id  
        size += 4; // data length
        size += p_type->data.size() * 4; // data
        size += 4; // processing_time_ms
        size += 8; // timestamp_ns

        if (payload->max_size < size)
        {
            payload->reserve(size);
        }

        payload->length = size;
        
        char* dst = reinterpret_cast<char*>(payload->data);
        size_t offset = 0;

        memcpy(dst + offset, &p_type->stream_id, 4);
        offset += 4;

        memcpy(dst + offset, &p_type->kernel_id, 4);
        offset += 4;

        uint32_t data_size = p_type->data.size();
        memcpy(dst + offset, &data_size, 4);
        offset += 4;

        if (data_size > 0)
        {
            memcpy(dst + offset, p_type->data.data(), data_size * 4);
            offset += data_size * 4;
        }

        memcpy(dst + offset, &p_type->processing_time_ms, 4);
        offset += 4;

        memcpy(dst + offset, &p_type->timestamp_ns, 8);

        return true;
    }

    bool deserialize(
            eprosima::fastrtps::rtps::SerializedPayload_t* payload,
            void* data) override
    {
        GpuProcessedData* p_type = static_cast<GpuProcessedData*>(data);
        
        char* src = reinterpret_cast<char*>(payload->data);
        size_t offset = 0;

        memcpy(&p_type->stream_id, src + offset, 4);
        offset += 4;

        memcpy(&p_type->kernel_id, src + offset, 4);
        offset += 4;

        uint32_t data_size = 0;
        memcpy(&data_size, src + offset, 4);
        offset += 4;

        p_type->data.resize(data_size);
        if (data_size > 0)
        {
            memcpy(p_type->data.data(), src + offset, data_size * 4);
            offset += data_size * 4;
        }

        memcpy(&p_type->processing_time_ms, src + offset, 4);
        offset += 4;

        memcpy(&p_type->timestamp_ns, src + offset, 8);

        return true;
    }

    std::function<uint32_t()> getSerializedSizeProvider(
            void* data) override
    {
        return [data]() -> uint32_t
        {
            GpuProcessedData* p = static_cast<GpuProcessedData*>(data);
            uint32_t size = 0;
            size += 4; // stream_id
            size += 4; // kernel_id
            size += 4; // data length
            size += p->data.size() * 4; // data
            size += 4; // processing_time_ms
            size += 8; // timestamp_ns
            return size;
        };
    }

    void* createData() override
    {
        return reinterpret_cast<void*>(new GpuProcessedData());
    }

    void deleteData(
            void* data) override
    {
        delete(reinterpret_cast<GpuProcessedData*>(data));
    }

    bool getKey(
            void* data,
            eprosima::fastrtps::rtps::InstanceHandle_t* handle,
            bool force_md5 = false) override
    {
        return false;
    }
};

#endif
