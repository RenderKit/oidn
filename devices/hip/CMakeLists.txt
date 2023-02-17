## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
endif()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# Find hip
find_package(hip)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Find MIOpen
find_package(miopen)

set(HIP_SOURCES
  hip_common.h
  hip_common.cpp
  hip_conv.h
  hip_conv.cpp
  hip_device.h
  hip_device.cpp
  hip_engine.h
  hip_engine.cpp
  hip_module.cpp
)

add_library(${PROJECT_NAME}_device_hip SHARED ${HIP_SOURCES} ${GPU_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME}_device_hip PROPERTY VERSION ${PROJECT_VERSION})
target_link_libraries(${PROJECT_NAME}_device_hip PRIVATE ${PROJECT_NAME}_core hip::device MIOpen)
oidn_strip_symbols(${PROJECT_NAME}_device_hip)
oidn_install_module(${PROJECT_NAME}_device_hip)