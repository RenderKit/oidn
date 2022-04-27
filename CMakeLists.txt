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
set(OIDN_VERSION_NOTE "-alpha")

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

# Library dependencies
find_package(Threads REQUIRED)
find_package(TBB 2017.0 REQUIRED tbb CONFIG)

# Packaging
include(oidn_package)

# Neural network runtime
set(OIDN_NEURAL_RUNTIMES)
if(OIDN_ARCH STREQUAL "X64")
  list(APPEND OIDN_NEURAL_RUNTIMES "DNNL")
endif()
if(APPLE AND OIDN_ARCH STREQUAL "ARM64")
  list(APPEND OIDN_NEURAL_RUNTIMES "BNNS")
endif()
list(GET OIDN_NEURAL_RUNTIMES 0 OIDN_NEURAL_RUNTIME_DEFAULT)
set(OIDN_NEURAL_RUNTIME ${OIDN_NEURAL_RUNTIME_DEFAULT} CACHE STRING "Neural network runtime to use.")
set_property(CACHE OIDN_NEURAL_RUNTIME PROPERTY STRINGS ${OIDN_NEURAL_RUNTIMES})

# DNNL
if(OIDN_NEURAL_RUNTIME STREQUAL "DNNL")
  option(OIDN_DEVICE_SYCL "Enable SYCL device." OFF)
  include(oidn_dnnl)
endif()

# Platform-specific settings
include(oidn_platform)

# BNNS
if(OIDN_NEURAL_RUNTIME STREQUAL "BNNS")
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

# CUDA
option(OIDN_DEVICE_CUDA "Enable CUDA device." OFF)
if(OIDN_DEVICE_CUDA)
  enable_language(CUDA)
  find_package(cuDNN REQUIRED)
endif()

# HIP
option(OIDN_DEVICE_HIP "Enable HIP device." OFF)
if(OIDN_DEVICE_HIP)
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
  ${OIDN_RESOURCE}
  core/cpu/cpu_autoexposure.h
  core/cpu/cpu_autoexposure.cpp
  core/cpu/cpu_device.h
  core/cpu/cpu_device.cpp
  core/cpu/cpu_image_copy.h
  core/cpu/cpu_image_copy.cpp
  core/cpu/cpu_input_process.h
  core/cpu/cpu_input_process.cpp
  core/cpu/cpu_output_process.h
  core/cpu/cpu_output_process.cpp
  core/cpu/cpu_upsample.h
  core/cpu/cpu_upsample.cpp
  core/gpu/gpu_autoexposure.h
  core/gpu/gpu_image_copy.h
  core/gpu/gpu_input_process.h
  core/gpu/gpu_output_process.h
  core/gpu/gpu_pool.h
  core/gpu/gpu_upsample.h
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

if(OIDN_NEURAL_RUNTIME STREQUAL "DNNL")
  list(APPEND CORE_SOURCES
    core/dnnl/dnnl_conv.h
    core/dnnl/dnnl_conv.cpp
    core/dnnl/dnnl_common.h
    core/dnnl/dnnl_common.cpp
    core/dnnl/dnnl_device.h
    core/dnnl/dnnl_device.cpp
    core/dnnl/dnnl_pool.h
    core/dnnl/dnnl_pool.cpp
    core/dnnl/dnnl_reorder.h
    core/dnnl/dnnl_tensor.h
    core/dnnl/dnnl_tensor.cpp
  )
elseif(OIDN_NEURAL_RUNTIME STREQUAL "BNNS")
  list(APPEND CORE_SOURCES
    core/bnns/bnns_common.h
    core/bnns/bnns_common.cpp
    core/bnns/bnns_conv.h
    core/bnns/bnns_conv.cpp
    core/bnns/bnns_device.h
    core/bnns/bnns_device.cpp
    core/bnns/bnns_pool.h
    core/bnns/bnns_pool.cpp
  )
endif()

if(OIDN_DEVICE_SYCL)
  set(CMAKE_CXX_STANDARD 17)

  list(APPEND CORE_SOURCES
    core/sycl/sycl_device.h
    core/sycl/sycl_device.cpp
    core/sycl/sycl_pool.h
    core/sycl/sycl_pool.cpp
    core/sycl/sycl_upsample.h
    core/sycl/sycl_upsample.cpp
  )

  set_source_files_properties(${CORE_SOURCES} PROPERTIES COMPILE_FLAGS "${OIDN_CXX_FLAGS_SYCL}")
endif()

if(OIDN_DEVICE_CUDA)
  list(APPEND CORE_SOURCES
    core/cuda/cuda_common.h
    core/cuda/cuda_common.cu
    core/cuda/cuda_conv.h
    core/cuda/cuda_conv.cu
    core/cuda/cuda_concat_conv.h
    core/cuda/cuda_concat_conv.cu
    core/cuda/cuda_device.h
    core/cuda/cuda_device.cu
  )
endif()

if(OIDN_DEVICE_HIP)
  list(APPEND CORE_SOURCES
    core/hip/hip_common.h
    core/hip/hip_common.cpp
    core/hip/hip_conv.h
    core/hip/hip_conv.cpp
    core/hip/hip_device.h
    core/hip/hip_device.cpp
  )
endif()

set(CORE_SOURCES_ISPC
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

add_library(${PROJECT_NAME} ${OIDN_LIB_TYPE} ${CORE_SOURCES})

target_compile_definitions(${PROJECT_NAME} PRIVATE OIDN_${OIDN_NEURAL_RUNTIME})
ispc_add_definitions(-DOIDN_${OIDN_NEURAL_RUNTIME})

ispc_target_add_sources(${PROJECT_NAME} ${CORE_SOURCES_ISPC})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME} PRIVATE common)
if(OIDN_NEURAL_RUNTIME STREQUAL "DNNL")
  target_link_libraries(${PROJECT_NAME} PRIVATE dnnl)
elseif(OIDN_NEURAL_RUNTIME STREQUAL "BNNS")
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Accelerate")
endif()

if(OIDN_DEVICE_SYCL)
  target_compile_definitions(${PROJECT_NAME} PUBLIC OIDN_DEVICE_SYCL)
  target_link_options(${PROJECT_NAME} PUBLIC "-fsycl")
endif()

if(OIDN_DEVICE_CUDA)
  target_compile_definitions(${PROJECT_NAME} PUBLIC OIDN_DEVICE_CUDA)
  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86")
  target_link_libraries(${PROJECT_NAME} PRIVATE cuDNN::cuDNN)
endif()

if(OIDN_DEVICE_HIP)
  target_compile_definitions(${PROJECT_NAME} PUBLIC OIDN_DEVICE_HIP)
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
