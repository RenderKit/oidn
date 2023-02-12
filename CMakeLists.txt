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

message(STATUS "Setting target version ${PROJECT_VERSION}")

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# CMake macros
include(oidn_macros)

# Detect the processor architecture
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Intel(R) Open Image Denoise supports 64-bit platforms only")
endif()
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
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
if(OIDN_API_NAMESPACE)
  set(OIDN_NAMESPACE ${OIDN_API_NAMESPACE}::oidn)
else()
  set(OIDN_NAMESPACE oidn)
endif()

# Output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Devices
option(OIDN_DEVICE_CPU "Enable CPU device." ON)
if(NOT APPLE)
  if(OIDN_ARCH STREQUAL "X64")
    option(OIDN_DEVICE_SYCL "Enable SYCL device." OFF)
  endif()
  option(OIDN_DEVICE_CUDA "Enable CUDA device." OFF)
  option(OIDN_DEVICE_HIP "Enable HIP device." OFF)
endif()

# Packaging
include(oidn_package)

# Platform-specific settings
include(oidn_platform)

# Generate list of symbols to export
configure_file(
  "${PROJECT_SOURCE_DIR}/common/export.linux.map.in"
  "${PROJECT_SOURCE_DIR}/common/export.linux.map"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/common/export.macos.map.in"
  "${PROJECT_SOURCE_DIR}/common/export.macos.map"
)

set(OIDN_RESOURCE ${PROJECT_SOURCE_DIR}/common/oidn.rc)

# Common static library
add_subdirectory(common)

## -----------------------------------------------------------------------------
## Core library
## -----------------------------------------------------------------------------

set(CORE_SOURCES
  core/autoexposure.h
  core/buffer.h
  core/buffer.cpp
  core/color.h
  core/color.cpp
  core/concat_conv.h
  core/concat_conv.cpp
  core/context.h
  core/context.cpp
  core/conv.cpp
  core/conv.h
  core/conv.cpp
  core/data.h
  core/device_factory.h
  core/device.h
  core/device.cpp
  core/engine.h
  core/engine.cpp
  core/exception.h
  core/exception.cpp
  core/filter.h
  core/filter.cpp
  core/image_accessor.h
  core/image_copy.h
  core/image.h
  core/image.cpp
  core/input_process.h
  core/input_process.cpp
  core/math.h
  core/module.h
  core/module.cpp
  core/network.h
  core/network.cpp
  core/op.h
  core/output_process.h
  core/output_process.cpp
  core/pool.h
  core/pool.cpp
  core/progress.h
  core/progress.cpp
  core/ref.h
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
  core/thread.h
  core/thread.cpp
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

set(GPU_SOURCES
  ${PROJECT_SOURCE_DIR}/core/gpu/gpu_autoexposure.h
  ${PROJECT_SOURCE_DIR}/core/gpu/gpu_image_copy.h
  ${PROJECT_SOURCE_DIR}/core/gpu/gpu_input_process.h
  ${PROJECT_SOURCE_DIR}/core/gpu/gpu_output_process.h
  ${PROJECT_SOURCE_DIR}/core/gpu/gpu_pool.h
  ${PROJECT_SOURCE_DIR}/core/gpu/gpu_upsample.h
)

add_library(${PROJECT_NAME}_core SHARED ${CORE_SOURCES})
set_property(TARGET ${PROJECT_NAME}_core PROPERTY VERSION ${PROJECT_VERSION})

# Include weights
option(OIDN_FILTER_RT "Include trained weights of the RT filter." ON)
option(OIDN_FILTER_RTLIGHTMAP "Include trained weights of the RTLightmap filter." ON)

set(WEIGHTS)

if(OIDN_FILTER_RT)
  target_compile_definitions(${PROJECT_NAME}_common PUBLIC OIDN_FILTER_RT)
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
  target_compile_definitions(${PROJECT_NAME}_common PUBLIC OIDN_FILTER_RTLIGHTMAP)
  list(APPEND WEIGHTS
    weights/rtlightmap_hdr.tza
    weights/rtlightmap_dir.tza
  )
endif()

oidn_generate_cpp_from_blob(WEIGHTS_SOURCES "${OIDN_NAMESPACE}::blobs::weights" ${WEIGHTS})
target_sources(${PROJECT_NAME}_core PRIVATE ${WEIGHTS_SOURCES})

target_include_directories(${PROJECT_NAME}_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  #PRIVATE # FIXME: why is this needed?
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME}_core PUBLIC ${PROJECT_NAME}_common)
oidn_export_all_symbols(${PROJECT_NAME}_core)

## -----------------------------------------------------------------------------
## Main library (API)
## -----------------------------------------------------------------------------

set(API_SOURCES
  core/api.cpp
)

add_library(${PROJECT_NAME} ${OIDN_LIB_TYPE} ${API_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION ${PROJECT_VERSION_MAJOR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_core)
oidn_strip_symbols(${PROJECT_NAME})

## -----------------------------------------------------------------------------
## Modules
## -----------------------------------------------------------------------------

if(OIDN_DEVICE_CPU)
  add_subdirectory(core/cpu)
endif()
if(OIDN_DEVICE_SYCL)
  add_subdirectory(core/sycl)
endif()
if(OIDN_DEVICE_CUDA)
  add_subdirectory(core/cuda)
endif()
if(OIDN_DEVICE_HIP)
  add_subdirectory(core/hip)
endif()

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

#include(oidn_install) # FIXME

# Has to be last
include(CPack)
