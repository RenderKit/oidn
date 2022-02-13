## Copyright 2009-2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.1)

# Policy: find_package() uses <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(OIDN_VERSION_MAJOR 1)
set(OIDN_VERSION_MINOR 4)
set(OIDN_VERSION_PATCH 3)
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

# CMake modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# CMake macros
include(oidn_macros)

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
find_package(TBB 2017.0 REQUIRED tbb)

# Platform-specific settings
include(oidn_platform)

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

if(OIDN_NEURAL_RUNTIME STREQUAL "DNNL")
  include(oidn_dnnl)
elseif(OIDN_NEURAL_RUNTIME STREQUAL "BNNS")
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
  core/conv.h
  core/cpu_buffer.h
  core/cpu_device.h
  core/cpu_device.cpp
  core/data.h
  core/device.h
  core/device.cpp
  core/filter.h
  core/filter.cpp
  core/image.h
  core/input_reorder.h
  core/input_reorder.cpp
  core/network.h
  core/network.cpp
  core/node.h
  core/output_copy.h
  core/output_copy.cpp
  core/output_reorder.h
  core/output_reorder.cpp
  core/pool.h
  core/progress.h
  core/reorder.h
  core/scratch.h
  core/scratch.cpp
  core/tensor.h
  core/tza.h
  core/tza.cpp
  core/unet.h
  core/unet.cpp
  core/upsample.h
  core/upsample.cpp
  core/vec.h
)

set(CORE_SOURCES_ISPC
  core/image.isph
  core/input_reorder.ispc
  core/color.isph
  core/color.ispc
  core/math.isph
  core/upsample.ispc
  core/output_copy.ispc
  core/output_reorder.ispc
  core/reorder.isph
  core/tensor.isph
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
