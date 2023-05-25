## Copyright 2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

# Set common paths
set(OIDN_ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Get the library version
include(${OIDN_ROOT_SOURCE_DIR}/cmake/oidn_version.cmake)

# Project
project(OpenImageDenoise_device_hip
  VERSION ${OIDN_VERSION}
  LANGUAGES CXX
)

# Set the CMake module path
list(APPEND CMAKE_MODULE_PATH ${OIDN_ROOT_SOURCE_DIR}/cmake)

# Common
include(oidn_common_external)

# Set the GPU targets
set(GPU_TARGETS "gfx1030,gfx1100,gfx1101,gfx1102" CACHE INTERNAL "")
set(AMDGPU_TARGETS ${GPU_TARGETS} CACHE INTERNAL "")

# Find HIP
find_package(hip REQUIRED)

set(OIDN_HIP_SOURCES
  ck_conv.h
  ck_conv_dl.cpp
  ck_conv_wmma.cpp
  hip_conv.h
  hip_device.h
  hip_device.cpp
  hip_engine.h
  hip_engine.cpp
  hip_external_buffer.h
  hip_external_buffer.cpp
  hip_module.cpp
)

add_library(OpenImageDenoise_device_hip SHARED ${OIDN_HIP_SOURCES} ${OIDN_GPU_SOURCES} ${OIDN_RESOURCE_FILE})
set_property(TARGET OpenImageDenoise_device_hip PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET OpenImageDenoise_device_hip PROPERTY CXX_STANDARD 17)

target_include_directories(OpenImageDenoise_device_hip
  PRIVATE
    "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/include"
    "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/library/include"
)

set_source_files_properties(
  ck_conv_wmma.cpp
  PROPERTIES COMPILE_FLAGS "-mcumode -mno-wavefrontsize64" # Navi3x optimizations
)

if(WIN32)
  # Fix warning: __declspec attribute 'dllexport' is not supported
  target_compile_options(OpenImageDenoise_device_hip PRIVATE -Wno-ignored-attributes)
endif()

target_link_libraries(OpenImageDenoise_device_hip PRIVATE OpenImageDenoise_core hip::device)
oidn_strip_symbols(OpenImageDenoise_device_hip)
oidn_install_module(OpenImageDenoise_device_hip)