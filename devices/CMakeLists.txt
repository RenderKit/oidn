## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(OIDN_GPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_autoexposure.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_image_copy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_input_process.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_output_process.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_pool.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_upsample.h
)

if(OIDN_DEVICE_CPU)
  add_subdirectory(cpu)
endif()
if(OIDN_DEVICE_SYCL)
  add_subdirectory(sycl)
endif()
if(OIDN_DEVICE_CUDA)
  add_subdirectory(cuda)
endif()
if(OIDN_DEVICE_HIP)
  add_subdirectory(hip)
endif()