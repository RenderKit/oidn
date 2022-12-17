## Copyright 2009-2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.1)

# Policy: find_package() uses <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(OIDN_VERSION_MAJOR 2)
set(OIDN_VERSION_MINOR 0)
set(OIDN_VERSION_PATCH 0)
set(OIDN_VERSION_NOTE "-beta")

set(OIDN_VERSION ${OIDN_VERSION_MAJOR}.${OIDN_VERSION_MINOR}.${OIDN_VERSION_PATCH})
math(EXPR OIDN_VERSION_NUMBER "10000*${OIDN_VERSION_MAJOR} + 100*${OIDN_VERSION_MINOR} + ${OIDN_VERSION_PATCH}")

project(OpenImageDenoise
  VERSION ${OIDN_VERSION}
  LANGUAGES C CXX
)

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# CMake macros
include(oidn_macros)

# Detect the processor architecture
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Intel(R) Open Image Denoise supports 64-bit platforms only")
endif()
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
  set(OIDN_ARCH "ARM64")
else()
  set(OIDN_ARCH "X64")
endif()

# Build as shared or static library
option(OIDN_STATIC_LIB "Build Intel(R) Open Image Denoise as a static library.")
mark_as_advanced(CLEAR OIDN_STATIC_LIB)
if(OIDN_STATIC_LIB)
  set(OIDN_LIB_TYPE STATIC)
else()
  set(OIDN_LIB_TYPE SHARED)
endif()

# Configuration types
set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
if(win32)
  if(NOT OIDN_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET)
    set(CMAKE_CONFIGURATION_TYPES "${CONFIGURATION_TYPES}"
        CACHE STRING "List of generated configurations." FORCE)
    set(OOIDN_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET ON
        CACHE INTERNAL "Default CMake configuration types set.")
  endif()
else()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CONFIGURATION_TYPES})
endif()

# API namespace
set(OIDN_API_NAMESPACE "" CACHE STRING "C++ namespace to put API symbols into.")

# Output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Packaging
include(oidn_package)

# CPU device
option(OIDN_DEVICE_CPU "Enable CPU device." ON)

# SYCL device
if(OIDN_ARCH STREQUAL "X64")
  option(OIDN_DEVICE_SYCL "Enable SYCL device." OFF)
endif()

if(OIDN_DEVICE_CPU)
  # TBB
  find_package(Threads REQUIRED)
  find_package(TBB 2017.0 REQUIRED tbb CONFIG)

  # DNNL
  if(OIDN_ARCH STREQUAL "X64")
    set(OIDN_DNNL ON)
    include(oidn_dnnl)
  endif()
endif()

# Platform-specific settings
include(oidn_platform)

# CPU device
if(OIDN_DEVICE_CPU)
  # BNNS
  if(APPLE AND OIDN_ARCH STREQUAL "ARM64")
    set(OIDN_BNNS ON)
    include(oidn_bnns)
  endif()

  # ISPC
  if(OIDN_ARCH STREQUAL "X64")
    set(OIDN_ISPC_TARGET_LIST sse4-i32x8;avx2-i32x8;avx512skx-i32x16)
  elseif(OIDN_ARCH STREQUAL "ARM64")
    set(OIDN_ISPC_TARGET_LIST neon-i32x8)
  endif()
  set(OIDN_ISPC_ADDRESSING 64)
  include(oidn_ispc)
endif()

# CUDA device
if(NOT APPLE)
  option(OIDN_DEVICE_CUDA "Enable CUDA device." OFF)
  if(OIDN_DEVICE_CUDA)
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
    if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.1")
      message(FATAL_ERROR "Building with CUDA support requires NVIDIA CUDA Toolkit 11.1 or newer")
    endif()
  endif()
endif()

# HIP device
if(NOT APPLE)
  option(OIDN_DEVICE_HIP "Enable HIP device." OFF)
  if(OIDN_DEVICE_HIP)
    # FIXME
    if(WIN32)
      message(FATAL_ERROR "The HIP device requires ROCm, which is not available on Windows yet.")
    endif()
    if(OIDN_DEVICE_SYCL OR OIDN_DEVICE_CUDA)
      message(FATAL_ERROR "Due to a CMake/ROCm bug, it is currently not possible to enable HIP with other GPU devices at the same time. This will be fixed in a future version.")
    endif()
    
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
  endif()
endif()

## -----------------------------------------------------------------------------
## Library
## -----------------------------------------------------------------------------

# Generate config.h
configure_file(
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h"
)

# Generate list of symbols to export
configure_file(
  "${PROJECT_SOURCE_DIR}/core/export.linux.map.in"
  "${PROJECT_SOURCE_DIR}/core/export.linux.map"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/core/export.macos.map.in"
  "${PROJECT_SOURCE_DIR}/core/export.macos.map"
)

set(OIDN_RESOURCE ${PROJECT_SOURCE_DIR}/common/oidn.rc)

add_subdirectory(common)

set(CORE_SOURCES
  include/OpenImageDenoise/oidn.h
  include/OpenImageDenoise/oidn.hpp
  include/OpenImageDenoise/config.h
  core/api.cpp
  core/autoexposure.h
  core/buffer.h
  core/buffer.cpp
  core/color.h
  core/color.cpp
  core/common.h
  core/common.cpp
  core/concat_conv.h
  core/concat_conv.cpp
  core/conv.cpp
  core/conv.h
  core/conv.cpp
  core/data.h
  core/device.h
  core/device.cpp
  core/engine.h
  core/engine.cpp
  core/filter.h
  core/filter.cpp
  core/image_accessor.h
  core/image_copy.h
  core/image.h
  core/image.cpp
  core/input_process.h
  core/input_process.cpp
  core/network.h
  core/network.cpp
  core/op.h
  core/output_process.h
  core/output_process.cpp
  core/pool.h
  core/pool.cpp
  core/progress.h
  core/progress.cpp
  core/reorder.h
  core/reorder.cpp
  core/rt_filter.h
  core/rt_filter.cpp
  core/rtlightmap_filter.h
  core/rtlightmap_filter.cpp
  core/scratch.h
  core/scratch.cpp
  core/tensor.h
  core/tensor.cpp
  core/tensor_accessor.h
  core/tensor_layout.h
  core/tile.h
  core/tza.h
  core/tza.cpp
  core/unet_filter.h
  core/unet_filter.cpp
  core/upsample.h
  core/upsample.cpp
  core/vec.h
  core/weights.h
  core/weights.cpp
)

if(OIDN_DEVICE_CPU)
  list(APPEND CORE_SOURCES
    core/cpu/cpu_autoexposure.h
    core/cpu/cpu_autoexposure.cpp
    core/cpu/cpu_device.h
    core/cpu/cpu_device.cpp
    core/cpu/cpu_engine.h
    core/cpu/cpu_engine.cpp
    core/cpu/cpu_image_copy.h
    core/cpu/cpu_image_copy.cpp
    core/cpu/cpu_input_process.h
    core/cpu/cpu_input_process.cpp
    core/cpu/cpu_output_process.h
    core/cpu/cpu_output_process.cpp
    core/cpu/cpu_upsample.h
    core/cpu/cpu_upsample.cpp
    core/cpu/tasking.h
    core/cpu/tasking.cpp
  )

  list(APPEND CORE_SOURCES_ISPC
    core/cpu/cpu_autoexposure.ispc
    core/cpu/cpu_image_copy.ispc
    core/cpu/cpu_input_process.ispc
    core/cpu/cpu_output_process.ispc
    core/cpu/cpu_upsample.ispc
    core/color.isph
    core/color.ispc
    core/image_accessor.isph
    core/math.isph
    core/tensor_accessor.isph
    core/tile.isph
    core/vec.isph
  )
endif()

if(OIDN_DEVICE_SYCL OR OIDN_DEVICE_CUDA OR OIDN_DEVICE_HIP)
  list(APPEND CORE_SOURCES
    core/gpu/gpu_autoexposure.h
    core/gpu/gpu_image_copy.h
    core/gpu/gpu_input_process.h
    core/gpu/gpu_output_process.h
    core/gpu/gpu_pool.h
    core/gpu/gpu_upsample.h
  )
endif()

if(OIDN_DNNL)
  list(APPEND CORE_SOURCES
    core/dnnl/dnnl_conv.h
    core/dnnl/dnnl_conv.cpp
    core/dnnl/dnnl_common.h
    core/dnnl/dnnl_common.cpp
    core/dnnl/dnnl_engine.h
    core/dnnl/dnnl_engine.cpp
    core/dnnl/dnnl_pool.h
    core/dnnl/dnnl_pool.cpp
    core/dnnl/dnnl_reorder.h
    core/dnnl/dnnl_tensor.h
    core/dnnl/dnnl_tensor.cpp
  )
elseif(OIDN_BNNS)
  list(APPEND CORE_SOURCES
    core/bnns/bnns_common.h
    core/bnns/bnns_common.cpp
    core/bnns/bnns_conv.h
    core/bnns/bnns_conv.cpp
    core/bnns/bnns_engine.h
    core/bnns/bnns_engine.cpp
    core/bnns/bnns_pool.h
    core/bnns/bnns_pool.cpp
  )
endif()

if(OIDN_DEVICE_SYCL)
  set(CMAKE_CXX_STANDARD 17)

  list(APPEND CORE_SOURCES
    core/sycl/sycl_common.h
    core/sycl/sycl_conv_gen.h
    core/sycl/sycl_conv_gen9.cpp
    core/sycl/sycl_conv_xehpc.cpp
    core/sycl/sycl_conv_xehpg.cpp
    core/sycl/sycl_conv.h
    core/sycl/sycl_device.h
    core/sycl/sycl_device.cpp
    core/sycl/sycl_engine.h
    core/sycl/sycl_engine.cpp
    core/sycl/sycl_external_buffer.h
    core/sycl/sycl_external_buffer.cpp
  )

  set_source_files_properties(${CORE_SOURCES} PROPERTIES COMPILE_FLAGS "${OIDN_CXX_FLAGS_SYCL}")
endif()

if(OIDN_DEVICE_CUDA)
  list(APPEND CORE_SOURCES
    core/cuda/cuda_conv.h
    core/cuda/cuda_conv.cu
    core/cuda/cuda_concat_conv.h
    core/cuda/cuda_concat_conv.cpp
    core/cuda/cuda_device.h
    core/cuda/cuda_device.cu
    core/cuda/cuda_engine.h
    core/cuda/cuda_engine.cu
    core/cuda/cuda_external_buffer.h
    core/cuda/cuda_external_buffer.cu
    core/cuda/cutlass_conv.h
    core/cuda/cutlass_conv_sm70.cu
    core/cuda/cutlass_conv_sm75.cu
    core/cuda/cutlass_conv_sm80.cu
  )

  # We do not generate PTX to avoid excessive JIT compile times
  set(OIDN_NVCC_SM70_FLAGS "-gencode arch=compute_70,code=sm_70")
  set(OIDN_NVCC_SM75_FLAGS "-gencode arch=compute_75,code=sm_75")
  set(OIDN_NVCC_SM80_FLAGS "-gencode arch=compute_80,code=sm_80")

  set_source_files_properties(
    core/cuda/cuda_conv.cu
    core/cuda/cuda_device.cu
    core/cuda/cuda_engine.cu
    PROPERTIES COMPILE_FLAGS
      "${OIDN_NVCC_SM70_FLAGS} ${OIDN_NVCC_SM75_FLAGS} ${OIDN_NVCC_SM80_FLAGS}"
  )

  set_source_files_properties(core/cuda/cutlass_conv_sm70.cu PROPERTIES COMPILE_FLAGS ${OIDN_NVCC_SM70_FLAGS})
  set_source_files_properties(core/cuda/cutlass_conv_sm75.cu PROPERTIES COMPILE_FLAGS ${OIDN_NVCC_SM75_FLAGS})
  set_source_files_properties(core/cuda/cutlass_conv_sm80.cu PROPERTIES COMPILE_FLAGS ${OIDN_NVCC_SM80_FLAGS})
endif()

if(OIDN_DEVICE_HIP)
  list(APPEND CORE_SOURCES
    core/hip/hip_common.h
    core/hip/hip_common.cpp
    core/hip/hip_conv.h
    core/hip/hip_conv.cpp
    core/hip/hip_device.h
    core/hip/hip_device.cpp
    core/hip/hip_engine.h
    core/hip/hip_engine.cpp
  )
endif()

add_library(${PROJECT_NAME} ${OIDN_LIB_TYPE} ${CORE_SOURCES} ${OIDN_RESOURCE})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE common)

if(OIDN_DNNL)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OIDN_DNNL)
  if(OIDN_DEVICE_CPU)
    ispc_add_definitions(-DOIDN_DNNL)
  endif()
  target_link_libraries(${PROJECT_NAME} PRIVATE dnnl)
elseif(OIDN_BNNS)
  target_compile_definitions(${PROJECT_NAME} PRIVATE OIDN_BNNS)
  ispc_add_definitions(-DOIDN_BNNS)
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Accelerate")
endif()

if(OIDN_DEVICE_CPU)
  ispc_target_add_sources(${PROJECT_NAME} ${CORE_SOURCES_ISPC})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_THREAD_LIBS_INIT} TBB::tbb)
endif()

if(OIDN_DEVICE_SYCL)
  target_link_options(${PROJECT_NAME} PRIVATE "-fsycl")
  target_link_libraries(${PROJECT_NAME} PRIVATE ze_loader)
endif()

if(OIDN_DEVICE_CUDA)
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/cutlass/include" "${PROJECT_SOURCE_DIR}/cutlass/tools/util/include")
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

if(OIDN_DEVICE_HIP)
  target_link_libraries(${PROJECT_NAME} PRIVATE hip::device MIOpen)
endif()

# Include weights
option(OIDN_FILTER_RT "Include trained weights of the RT filter." ON)
option(OIDN_FILTER_RTLIGHTMAP "Include trained weights of the RTLightmap filter." ON)

set(WEIGHTS)

if(OIDN_FILTER_RT)
  target_compile_definitions(${PROJECT_NAME} PUBLIC OIDN_FILTER_RT)
  list(APPEND WEIGHTS
    weights/rt_alb.tza
    weights/rt_hdr.tza
    weights/rt_hdr_alb.tza
    weights/rt_hdr_alb_nrm.tza
    weights/rt_hdr_calb_cnrm.tza
    weights/rt_ldr.tza
    weights/rt_ldr_alb.tza
    weights/rt_ldr_alb_nrm.tza
    weights/rt_ldr_calb_cnrm.tza
    weights/rt_nrm.tza
  )
endif()

if(OIDN_FILTER_RTLIGHTMAP)
  target_compile_definitions(${PROJECT_NAME} PUBLIC OIDN_FILTER_RTLIGHTMAP)
  list(APPEND WEIGHTS
    weights/rtlightmap_hdr.tza
    weights/rtlightmap_dir.tza
  )
endif()

generate_cpp_from_blob(WEIGHTS_SOURCES "oidn::blobs::weights" ${WEIGHTS})
target_sources(${PROJECT_NAME} PRIVATE ${WEIGHTS_SOURCES})

# Specify list of symbols to export
if(WIN32)
elseif(APPLE)
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS -Wl,-exported_symbols_list,${PROJECT_SOURCE_DIR}/core/export.macos.map)
  set_source_files_properties(core/api.cpp PROPERTIES OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/core/export.macos.map) 
else()
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS -Wl,--version-script=${PROJECT_SOURCE_DIR}/core/export.linux.map)
  set_source_files_properties(core/api.cpp PROPERTIES OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/core/export.linux.map)
endif()

message(STATUS "Setting target version ${PROJECT_VERSION}")
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION ${PROJECT_VERSION_MAJOR})

## -----------------------------------------------------------------------------
## Apps
## -----------------------------------------------------------------------------

option(OIDN_APPS "Enable building example and test apps." ON)
if(OIDN_APPS)
  add_subdirectory(apps)
endif()

## -----------------------------------------------------------------------------
## Install and packaging
## -----------------------------------------------------------------------------

include(oidn_install)

# Has to be last
include(CPack)
