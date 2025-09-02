#!/bin/bash
# Generar tipos desde IDL
docker run --rm -v $(pwd):/data fastdds:gpu-fixed bash -c "
cd /data
fastddsgen -example CMake idl/GpuData.idl
"
