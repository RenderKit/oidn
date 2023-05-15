## Copyright 2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  cmake_minimum_required(VERSION 3.23.2)
else()
  cmake_minimum_required(VERSION 3.18)
endif()

# Set common paths
set(OIDN_ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Get the library version
include(${OIDN_ROOT_SOURCE_DIR}/cmake/oidn_version.cmake)

# Project
project(OpenImageDenoise_device_cuda
  VERSION ${OIDN_VERSION}
  LANGUAGES CXX CUDA
)

# Check the CUDA compiler
set(OIDN_CUDA_MIN_VERSION 11.8)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS ${OIDN_CUDA_MIN_VERSION})
  message(FATAL_ERROR "Building with CUDA support requires NVIDIA CUDA Toolkit ${OIDN_CUDA_MIN_VERSION} or newer")
endif()

# Set the CMake module path
list(APPEND CMAKE_MODULE_PATH ${OIDN_ROOT_SOURCE_DIR}/cmake)

# Common
include(oidn_common_external)

set(OIDN_CUDA_SOURCES
  cuda_conv.h
  cuda_conv.cu
  cuda_device.h
  cuda_device.cpp
  cuda_engine.h
  cuda_engine.cu
  cuda_external_buffer.h
  cuda_external_buffer.cpp
  cuda_module.cpp
  cutlass_conv.h
  cutlass_conv_sm70.cu
  cutlass_conv_sm75.cu
  cutlass_conv_sm80.cu
)

# We do not generate PTX to avoid excessive JIT compile times
set(OIDN_NVCC_SM70_FLAGS "-gencode arch=compute_70,code=sm_70")
set(OIDN_NVCC_SM75_FLAGS "-gencode arch=compute_75,code=sm_75")
set(OIDN_NVCC_SM80_FLAGS "-gencode arch=compute_80,code=sm_80")
set(OIDN_NVCC_SM90_FLAGS "-gencode arch=compute_90,code=sm_90")

set_source_files_properties(
  cuda_conv.cu
  cuda_device.cu
  cuda_engine.cu
  PROPERTIES COMPILE_FLAGS
    "${OIDN_NVCC_SM70_FLAGS} ${OIDN_NVCC_SM75_FLAGS} ${OIDN_NVCC_SM80_FLAGS} ${OIDN_NVCC_SM90_FLAGS}"
)

set_source_files_properties(
  cutlass_conv_sm70.cu
  PROPERTIES COMPILE_FLAGS ${OIDN_NVCC_SM70_FLAGS})

set_source_files_properties(
  cutlass_conv_sm75.cu
  PROPERTIES COMPILE_FLAGS ${OIDN_NVCC_SM75_FLAGS})

set_source_files_properties(
  cutlass_conv_sm80.cu
  PROPERTIES COMPILE_FLAGS "${OIDN_NVCC_SM80_FLAGS} ${OIDN_NVCC_SM90_FLAGS}")

add_library(OpenImageDenoise_device_cuda SHARED ${OIDN_CUDA_SOURCES} ${OIDN_GPU_SOURCES} ${OIDN_RESOURCE_FILE})
set_property(TARGET OpenImageDenoise_device_cuda PROPERTY VERSION ${PROJECT_VERSION})

target_include_directories(OpenImageDenoise_device_cuda
  PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    "${OIDN_ROOT_SOURCE_DIR}/external/cutlass/include"
    "${OIDN_ROOT_SOURCE_DIR}/external/cutlass/tools/util/include")
set_target_properties(OpenImageDenoise_device_cuda PROPERTIES CUDA_ARCHITECTURES OFF)

target_link_libraries(OpenImageDenoise_device_cuda PRIVATE OpenImageDenoise_core)
oidn_strip_symbols(OpenImageDenoise_device_cuda)
oidn_install_module(OpenImageDenoise_device_cuda)