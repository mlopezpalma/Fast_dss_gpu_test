// gpu_dds_types.h - Data types for GPU-DDS communication

#ifndef GPU_DDS_TYPES_H
#define GPU_DDS_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace gpu_dds {

// GPU processing result
struct GpuProcessedData {
    uint32_t stream_id;
    uint32_t kernel_id;
    std::string kernel_name;
    
    // Timing metrics
    float processing_time_ms;
    float throughput_gbps;
    
    // Data
    std::vector<float> input_data;    // Original data
    std::vector<float> output_data;   // Processed data
    
    // Metadata
    uint64_t timestamp_ns;
    uint32_t data_size;
    
    GpuProcessedData() 
        : stream_id(0)
        , kernel_id(0)
        , processing_time_ms(0.0f)
        , throughput_gbps(0.0f)
        , timestamp_ns(0)
        , data_size(0) {}
};

// Performance metrics
struct GpuMetrics {
    uint32_t total_kernels_executed;
    float average_time_ms;
    float average_throughput_gbps;
    float gpu_memory_used_mb;
    float gpu_utilization_percent;
};

} // namespace gpu_dds

#endif // GPU_DDS_TYPES_H
