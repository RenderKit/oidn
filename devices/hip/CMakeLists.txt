## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
endif()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# Set the GPU targets
set(GPU_TARGETS "gfx908,gfx90a,gfx1030,gfx1100,gfx1101,gfx1102" CACHE INTERNAL "")
set(AMDGPU_TARGETS ${GPU_TARGETS} CACHE INTERNAL "")

# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

set(HIP_SOURCES
  ck_conv.h
  ck_conv_dl.cpp
  ck_conv_wmma.cpp
  ck_conv_xdl.cpp
  hip_conv.h
  hip_device.h
  hip_device.cpp
  hip_engine.h
  hip_engine.cpp
  hip_external_buffer.h
  hip_external_buffer.cpp
  hip_module.cpp
)

add_library(${PROJECT_NAME}_device_hip SHARED ${HIP_SOURCES} ${GPU_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME}_device_hip PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME}_device_hip PROPERTY CXX_STANDARD 17)

target_include_directories(${PROJECT_NAME}_device_hip
  PRIVATE
    "${PROJECT_SOURCE_DIR}/external/composable_kernel/include"
    "${PROJECT_SOURCE_DIR}/external/composable_kernel/library/include"
)

target_link_libraries(${PROJECT_NAME}_device_hip PRIVATE ${PROJECT_NAME}_core hip::device)
oidn_strip_symbols(${PROJECT_NAME}_device_hip)
oidn_install_module(${PROJECT_NAME}_device_hip)