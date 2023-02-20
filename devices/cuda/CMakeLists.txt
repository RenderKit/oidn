## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(CMAKE_GENERATOR MATCHES "Visual Studio" AND ${CMAKE_VERSION} VERSION_LESS "3.23.2")
  message(FATAL_ERROR "Building with CUDA support requires CMake 3.23.2 or newer if using a Visual Studio generator or 3.18.0 or newer otherwise (e.g. if using Ninja)")
elseif(${CMAKE_VERSION} VERSION_LESS "3.18.0")
  message(FATAL_ERROR "Building with CUDA support requires CMake 3.18.0 or newer")
endif()

# Set the NVCC host compiler to make sure the same C++ compiler is used for all host code (except DPC++, which is not supported yet)
if(NOT WIN32 AND (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15))
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

enable_language(CUDA)
if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.8")
  message(FATAL_ERROR "Building with CUDA support requires NVIDIA CUDA Toolkit 11.8 or newer")
endif()

set(CUDA_SOURCES
  cuda_conv.h
  cuda_conv.cu
  cuda_device.h
  cuda_device.cu
  cuda_engine.h
  cuda_engine.cu
  cuda_external_buffer.h
  cuda_external_buffer.cu
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

add_library(${PROJECT_NAME}_device_cuda SHARED ${CUDA_SOURCES} ${GPU_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME}_device_cuda PROPERTY VERSION ${PROJECT_VERSION})

target_include_directories(${PROJECT_NAME}_device_cuda
  PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    "${PROJECT_SOURCE_DIR}/external/cutlass/include"
    "${PROJECT_SOURCE_DIR}/external/cutlass/tools/util/include")
set_target_properties(${PROJECT_NAME}_device_cuda PROPERTIES CUDA_ARCHITECTURES OFF)

target_link_libraries(${PROJECT_NAME}_device_cuda PRIVATE ${PROJECT_NAME}_core)
oidn_strip_symbols(${PROJECT_NAME}_device_cuda)
oidn_install_module(${PROJECT_NAME}_device_cuda)