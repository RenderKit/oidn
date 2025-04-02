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
  LANGUAGES CXX
)

# CUDA
find_package(CUDAToolkit 12.8 REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
enable_language(CUDA)

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
function(oidn_set_cuda_sm_flags output_var)
  set(${output_var} "")
  foreach(version IN LISTS ARGN)
    list(APPEND flags "-gencode arch=compute_${version},code=sm_${version}")
  endforeach()
  string(JOIN " " ${output_var} ${flags})
  set(${output_var} "${${output_var}}" PARENT_SCOPE)
endfunction()

oidn_set_cuda_sm_flags(OIDN_CUDA_SM_FLAGS 70 75 80 90 100 120)
set_source_files_properties(
  cuda_conv.cu
  cuda_device.cu
  cuda_engine.cu
  PROPERTIES COMPILE_FLAGS ${OIDN_CUDA_SM_FLAGS}
)

oidn_set_cuda_sm_flags(OIDN_CUDA_SM70_FLAGS 70)
set_source_files_properties(
  cutlass_conv_sm70.cu
  PROPERTIES COMPILE_FLAGS ${OIDN_CUDA_SM70_FLAGS}
)

oidn_set_cuda_sm_flags(OIDN_CUDA_SM75_FLAGS 75)
set_source_files_properties(
  cutlass_conv_sm75.cu
  PROPERTIES COMPILE_FLAGS ${OIDN_CUDA_SM75_FLAGS}
)

oidn_set_cuda_sm_flags(OIDN_CUDA_SM80_FLAGS 80 90 100 120)
set_source_files_properties(
  cutlass_conv_sm80.cu
  PROPERTIES COMPILE_FLAGS ${OIDN_CUDA_SM80_FLAGS}
)

add_library(OpenImageDenoise_device_cuda SHARED ${OIDN_CUDA_SOURCES} ${OIDN_GPU_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_device_cuda PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_cuda
  CXX_STANDARD 17
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_cuda PROPERTIES VERSION ${PROJECT_VERSION})
endif()

target_include_directories(OpenImageDenoise_device_cuda
  PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    "${OIDN_ROOT_SOURCE_DIR}/external/cutlass/include"
    "${OIDN_ROOT_SOURCE_DIR}/external/cutlass/tools/util/include"
)

if(OIDN_DEVICE_CUDA_API STREQUAL "Driver")
  add_library(curtn STATIC curtn.cpp)
  target_link_libraries(curtn PUBLIC CUDA::cuda_driver)

  target_compile_definitions(OpenImageDenoise_device_cuda PRIVATE OIDN_DEVICE_CUDA_API_DRIVER)
  target_link_libraries(OpenImageDenoise_device_cuda PRIVATE curtn)

  set(OIDN_CUDA_RUNTIME_LIBRARY "None")
elseif(OIDN_DEVICE_CUDA_API STREQUAL "RuntimeStatic")
  set(OIDN_CUDA_RUNTIME_LIBRARY "Static")
elseif(OIDN_DEVICE_CUDA_API STREQUAL "RuntimeShared")
  set(OIDN_CUDA_RUNTIME_LIBRARY "Shared")
else()
  message(FATAL_ERROR "Invalid OIDN_DEVICE_CUDA_API value")
endif()

set_target_properties(OpenImageDenoise_device_cuda PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_RUNTIME_LIBRARY ${OIDN_CUDA_RUNTIME_LIBRARY}
)

# Fix warning: support for offline compilation for architectures prior to SM75 will be removed
# in a future release
target_compile_options(OpenImageDenoise_device_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
# Fix warning: calling a constexpr __host__ function from a __host__ __device__ function is
# not allowed (CUTLASS)
target_compile_options(OpenImageDenoise_device_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(OpenImageDenoise_device_cuda PRIVATE OpenImageDenoise_core)
oidn_strip_symbols(OpenImageDenoise_device_cuda)
oidn_install_module(OpenImageDenoise_device_cuda)