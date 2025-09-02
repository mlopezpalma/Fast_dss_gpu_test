cmake_minimum_required(VERSION 3.18)
project(FastDDS_GPU_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Set paths for Fast-DDS installed from source
set(CMAKE_PREFIX_PATH "/usr/local")
list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/fastdds")

# Find packages with correct names
find_package(fastcdr REQUIRED)
find_package(fastrtps REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
include_directories(${CUDA_INCLUDE_DIRS})

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# GPU Publisher with CUDA
add_executable(gpu_dds_publisher
    src/gpu_dds_publisher.cpp
    src/gpu_kernels.cu
)

set_target_properties(gpu_dds_publisher PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(gpu_dds_publisher
    fastrtps
    fastcdr
    ${CUDA_LIBRARIES}
    cudart
    cuda
)

# GPU Subscriber
add_executable(gpu_dds_subscriber
    src/gpu_dds_subscriber.cpp
)

target_link_libraries(gpu_dds_subscriber
    fastrtps
    fastcdr
)

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Nueva versión del subscriber
add_executable(gpu_dds_subscriber_v2
    src/gpu_dds_subscriber_v2.cpp
)

target_link_libraries(gpu_dds_subscriber_v2
    fastrtps
    fastcdr
    foonathan_memory
    pthread
)

# Publisher V2 con tipos correctos
add_executable(gpu_dds_publisher_v2
    src/gpu_dds_publisher_v2.cpp
    src/gpu_kernels.cu
)

set_target_properties(gpu_dds_publisher_v2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(gpu_dds_publisher_v2
    fastrtps
    fastcdr
    foonathan_memory
    ${CUDA_LIBRARIES}
    cudart
    cuda
    pthread
)

# Debug subscriber
add_executable(gpu_dds_subscriber_debug
    src/gpu_dds_subscriber_debug.cpp
)

target_link_libraries(gpu_dds_subscriber_debug
    fastrtps
    fastcdr
    foonathan_memory
    pthread
)
