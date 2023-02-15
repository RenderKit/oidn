## Copyright 2009-2023 Intel Corporation
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

# Filters
option(OIDN_FILTER_RT "Include trained weights of the RT filter." ON)
option(OIDN_FILTER_RTLIGHTMAP "Include trained weights of the RTLightmap filter." ON)

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

## -----------------------------------------------------------------------------
## Weights
## -----------------------------------------------------------------------------

set(WEIGHTS)

if(OIDN_FILTER_RT)
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
  list(APPEND WEIGHTS
    weights/rtlightmap_hdr.tza
    weights/rtlightmap_dir.tza
  )
endif()

oidn_generate_cpp_from_blob(WEIGHTS_SOURCES "${OIDN_NAMESPACE}::blobs::weights" ${WEIGHTS})
add_library(${PROJECT_NAME}_weights STATIC ${WEIGHTS_SOURCES})

## -----------------------------------------------------------------------------
## Libraries
## -----------------------------------------------------------------------------

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(api)
add_subdirectory(devices)

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
