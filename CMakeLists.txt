## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.1)

# Policy: find_package() uses <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(OIDN_VERSION_MAJOR 1)
set(OIDN_VERSION_MINOR 2)
set(OIDN_VERSION_PATCH 4)
set(OIDN_VERSION_NOTE "")

set(OIDN_VERSION ${OIDN_VERSION_MAJOR}.${OIDN_VERSION_MINOR}.${OIDN_VERSION_PATCH})
math(EXPR OIDN_VERSION_NUMBER "10000*${OIDN_VERSION_MAJOR} + 100*${OIDN_VERSION_MINOR} + ${OIDN_VERSION_PATCH}")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(OpenImageDenoise
  VERSION ${OIDN_VERSION}
  LANGUAGES C CXX
)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Intel(R) Open Image Denoise supports 64-bit platforms only")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# CMake macros
include(oidn_macros)


option(OIDN_BNNS "use Accelrate's framework for convolutions and pooling layers" OFF)
option(APPLE_SILICON "build for apple silicon ARM based products, using Accelerate framework for NN operations" OFF)

if (APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
  set(APPLE_SILICON ON)
  set(OIDN_BNNS ON)
  message(STATUS "Building OIDN library using BNNS API in Accelerate framework For Apple Silicon")
endif()




if (APPLE AND OIDN_BNNS)
  # check for minimal Apple SDK requirements

  if (NOT IOS)
    set(SDK_VERSION_COMMAND  xcrun -sdk  macosx --show-sdk-version)
    set(SDK_TARGET 11.0)
  else()
    set(SDK_VERSION_COMMAND  xcrun -sdk  iphoneos --show-sdk-version)
    set(SDK_TARGET 14.0)
  endif()
  execute_process(COMMAND  ${SDK_VERSION_COMMAND}
                  OUTPUT_VARIABLE SDK_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (SDK_VERSION VERSION_LESS SDK_TARGET)
    message(FATAL_ERROR "Needs Apple SDK version " ${SDK_TARGET} " or above to use BNNS, current SDK version " ${SDK_VERSION})
  endif()
  add_definitions(-DUSE_BNNS)
  add_link_options(-framework Accelerate)
  if (CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS SDK_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 11.0)
  endif()
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
find_package(TBB 2017.0 REQUIRED tbb tbbmalloc)

# Packaging
include(oidn_package)

# Platform-specific settings
include(oidn_platform)

# DNNL
include(oidn_dnnl)

# ISPC
if (NOT APPLE_SILICON)
  set(OIDN_ISPC_TARGET_LIST sse4-i32x8;avx2-i32x8;avx512skx-i32x16)
else()
  set(OIDN_ISPC_TARGET_LIST neon-i32x8)
endif()

set(OIDN_ISPC_ADDRESSING 64)
include(oidn_ispc)

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
  core/api.cpp
  core/buffer.h
  core/color.h
  core/color.cpp
  core/common.h
  core/data.h
  core/device.h
  core/device.cpp
  core/filter.h
  core/filter.cpp
  core/image.h
  core/input_reorder.h
  core/memory.h
  core/network.h
  core/network.cpp
  core/node.h
  core/output_copy.h
  core/output_reorder.h
  core/progress.h
  core/unet.h
  core/unet.cpp
  core/upsample.h
)

if (OIDN_BNNS)
  set(CORE_SOURCES ${CORE_SOURCES}
     bnns/bnns_network.h
     bnns/bnns_network.cpp
     bnns/bnns_node.h
  )
endif()

set(CORE_SOURCES_ISPC
  core/image.ih
  core/input_reorder.ispc
  core/color.ih
  core/color.ispc
  core/math.ih
  core/memory.ih
  core/upsample.ispc
  core/output_copy.ispc
  core/output_reorder.ispc
  core/vec.ih
)

generate_cpp_from_blob(WEIGHTS_SOURCES "oidn::blobs::weights"
  weights/rt_ldr.tza
  weights/rt_ldr_alb.tza
  weights/rt_ldr_alb_nrm.tza
  weights/rt_hdr.tza
  weights/rt_hdr_alb.tza
  weights/rt_hdr_alb_nrm.tza
  weights/rtlightmap_hdr.tza
)

add_library(${PROJECT_NAME} ${OIDN_LIB_TYPE} ${CORE_SOURCES} ${WEIGHTS_SOURCES})
ispc_target_add_sources(${PROJECT_NAME} ${CORE_SOURCES_ISPC})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    common
    dnnl
)

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
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION "0")

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
