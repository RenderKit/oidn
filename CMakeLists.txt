## ======================================================================== ##
## Copyright 2009-2020 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

cmake_minimum_required(VERSION 3.1)

# Policy: find_package() uses <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(OIDN_VERSION_MAJOR 1)
set(OIDN_VERSION_MINOR 2)
set(OIDN_VERSION_PATCH 0)
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

# Output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Library dependencies
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

# Packaging
include(oidn_package)

# Platform-specific settings
include(oidn_platform)

## -----------------------------------------------------------------------------
## DNNL
## -----------------------------------------------------------------------------

set(DNNL_VERSION_MAJOR 1)
set(DNNL_VERSION_MINOR 2)
set(DNNL_VERSION_PATCH 1)
set(DNNL_VERSION_HASH  "N/A")

set(DNNL_CPU_RUNTIME "TBB")
set(DNNL_CPU_THREADING_RUNTIME "TBB")
set(DNNL_GPU_RUNTIME "NONE")

configure_file(
  "${PROJECT_SOURCE_DIR}/mkl-dnn/include/dnnl_config.h.in"
  "${PROJECT_BINARY_DIR}/mkl-dnn/include/dnnl_config.h"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/mkl-dnn/include/dnnl_version.h.in"
  "${PROJECT_BINARY_DIR}/mkl-dnn/include/dnnl_version.h"
)

file(GLOB_RECURSE DNNL_COMMON_SOURCES
  mkl-dnn/src/common/*.h
  mkl-dnn/src/common/*.hpp
  mkl-dnn/src/common/*.c
  mkl-dnn/src/common/*.cpp
)

add_library(dnnl_common STATIC ${DNNL_COMMON_SOURCES})

target_include_directories(dnnl_common
  PUBLIC
    ${PROJECT_SOURCE_DIR}/mkl-dnn/include
    ${PROJECT_BINARY_DIR}/mkl-dnn/include
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src/common
)

target_compile_definitions(dnnl_common
  PUBLIC
    -DDNNL_ENABLE_CONCURRENT_EXEC
)

set(DNNL_COMPILE_OPTIONS ${OIDN_ISA_FLAGS_SSE41})
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # Correct 'jnl' macro/jit issue
  list(APPEND DNNL_COMPILE_OPTIONS "/Qlong-double")
endif()
target_compile_options(dnnl_common PRIVATE ${DNNL_COMPILE_OPTIONS})

target_link_libraries(dnnl_common PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${TBB_LIBRARIES})

## -----------------------------------------------------------------------------
## DNNL: CPU
## -----------------------------------------------------------------------------

file(GLOB_RECURSE DNNL_CPU_SOURCES
  mkl-dnn/src/cpu/bfloat16.cpp
  mkl-dnn/src/cpu/cpu_barrier.hpp
  mkl-dnn/src/cpu/cpu_barrier.cpp
  mkl-dnn/src/cpu/cpu_concat.cpp
  mkl-dnn/src/cpu/cpu_concat_pd.hpp
  mkl-dnn/src/cpu/cpu_convolution_list.cpp
  mkl-dnn/src/cpu/cpu_convolution_pd.hpp
  mkl-dnn/src/cpu/cpu_engine.hpp
  mkl-dnn/src/cpu/cpu_engine.cpp
  mkl-dnn/src/cpu/cpu_isa_traits.hpp
  mkl-dnn/src/cpu/cpu_isa_traits.cpp
  mkl-dnn/src/cpu/cpu_memory_storage.hpp
  mkl-dnn/src/cpu/cpu_pooling_list.cpp
  mkl-dnn/src/cpu/cpu_pooling_pd.hpp
  mkl-dnn/src/cpu/cpu_primitive.hpp
  mkl-dnn/src/cpu/cpu_reducer.hpp
  mkl-dnn/src/cpu/cpu_reducer.cpp
  mkl-dnn/src/cpu/cpu_reorder.cpp
  mkl-dnn/src/cpu/cpu_reorder_pd.hpp
  mkl-dnn/src/cpu/cpu_stream.hpp
  mkl-dnn/src/cpu/cpu_sum.cpp
  mkl-dnn/src/cpu/jit_avx2_conv_kernel_f32.hpp
  mkl-dnn/src/cpu/jit_avx2_conv_kernel_f32.cpp
  mkl-dnn/src/cpu/jit_avx2_convolution.hpp
  mkl-dnn/src/cpu/jit_avx2_convolution.cpp
  mkl-dnn/src/cpu/jit_avx512_common_conv_kernel.hpp
  mkl-dnn/src/cpu/jit_avx512_common_conv_kernel.cpp
  mkl-dnn/src/cpu/jit_avx512_common_convolution.hpp
  mkl-dnn/src/cpu/jit_avx512_common_convolution.cpp
  mkl-dnn/src/cpu/jit_avx512_common_convolution_winograd.hpp
  mkl-dnn/src/cpu/jit_avx512_common_convolution_winograd.cpp
  mkl-dnn/src/cpu/jit_avx512_common_conv_winograd_kernel_f32.hpp
  mkl-dnn/src/cpu/jit_avx512_common_conv_winograd_kernel_f32.cpp
  mkl-dnn/src/cpu/jit_avx512_core_f32_wino_conv_*.hpp
  mkl-dnn/src/cpu/jit_avx512_core_f32_wino_conv_*.cpp
  mkl-dnn/src/cpu/jit_generator.hpp
  mkl-dnn/src/cpu/jit_primitive_conf.hpp
  mkl-dnn/src/cpu/jit_sse41_conv_kernel_f32.hpp
  mkl-dnn/src/cpu/jit_sse41_conv_kernel_f32.cpp
  mkl-dnn/src/cpu/jit_sse41_convolution.hpp
  mkl-dnn/src/cpu/jit_sse41_convolution.cpp
  mkl-dnn/src/cpu/jit_transpose_src_utils.hpp
  mkl-dnn/src/cpu/jit_transpose_src_utils.cpp
  mkl-dnn/src/cpu/jit_uni_eltwise.hpp
  mkl-dnn/src/cpu/jit_uni_eltwise.cpp
  mkl-dnn/src/cpu/jit_uni_eltwise_injector.hpp
  mkl-dnn/src/cpu/jit_uni_eltwise_injector.cpp
  mkl-dnn/src/cpu/jit_uni_pooling.hpp
  mkl-dnn/src/cpu/jit_uni_pooling.cpp
  mkl-dnn/src/cpu/jit_uni_pool_kernel.hpp
  mkl-dnn/src/cpu/jit_uni_pool_kernel.cpp
  mkl-dnn/src/cpu/jit_uni_reorder.hpp
  mkl-dnn/src/cpu/jit_uni_reorder.cpp
  mkl-dnn/src/cpu/jit_uni_reorder_utils.cpp
  mkl-dnn/src/cpu/simple_q10n.hpp
  mkl-dnn/src/cpu/simple_reorder.hpp
  mkl-dnn/src/cpu/wino_reorder.hpp
  mkl-dnn/src/cpu/resampling/cpu_resampling_list.cpp
  mkl-dnn/src/cpu/resampling/cpu_resampling_pd.hpp
  mkl-dnn/src/cpu/resampling/resampling_utils.hpp
  mkl-dnn/src/cpu/resampling/simple_resampling.hpp
  mkl-dnn/src/cpu/resampling/simple_resampling.cpp
  mkl-dnn/src/cpu/jit_utils/*.h
  mkl-dnn/src/cpu/jit_utils/*.hpp
  mkl-dnn/src/cpu/jit_utils/*.c
  mkl-dnn/src/cpu/jit_utils/*.cpp
  mkl-dnn/src/cpu/xbyak/*.h
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  file(GLOB DNNL_CPU_SOURCES_BIGOBJ
    mkl-dnn/src/cpu/cpu_engine.cpp
    mkl-dnn/src/cpu/cpu_reorder.cpp
  )
  set_source_files_properties(${DNNL_CPU_SOURCES_BIGOBJ} PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

add_library(dnnl_cpu STATIC ${DNNL_CPU_SOURCES})

target_include_directories(dnnl_cpu
  PUBLIC
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu/xbyak
)

target_compile_options(dnnl_cpu PRIVATE ${DNNL_COMPILE_OPTIONS})

target_link_libraries(dnnl_cpu PUBLIC dnnl_common)

## -----------------------------------------------------------------------------
## OIDN
## -----------------------------------------------------------------------------

# ISPC
set(OIDN_ISPC_TARGET_LIST sse4-i32x8;avx2-i32x8;avx512skx-i32x16;avx512knl-i32x16)
set(OIDN_ISPC_ADDRESSING 64)
include(oidn_ispc)

# Generate version.h
configure_file(
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/version.h.in"
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/version.h"
)

add_subdirectory(common EXCLUDE_FROM_ALL)

set(CORE_SOURCES
  include/OpenImageDenoise/oidn.h
  include/OpenImageDenoise/oidn.hpp
  include/OpenImageDenoise/version.h
  core/autoencoder.h
  core/autoencoder.cpp
  core/api.cpp
  core/buffer.h
  core/color.h
  core/color.cpp
  core/common.h
  core/device.h
  core/device.cpp
  core/filter.h
  core/filter.cpp
  core/image.h
  core/input_reorder.h
  core/math.h
  core/memory.h
  core/network.h
  core/network.cpp
  core/node.h
  core/output_reorder.h
  core/upsample.h
)

set(CORE_SOURCES_ISPC
  core/image.ih
  core/input_reorder.ispc
  core/color.ih
  core/color.ispc
  core/math.ih
  core/memory.ih
  core/upsample.ispc
  core/output_reorder.ispc
  core/vec.ih
)

generate_cpp_resources(WEIGHTS_SOURCES "oidn::weights"
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

if(OIDN_STATIC_LIB)
  target_compile_definitions(${PROJECT_NAME} INTERFACE -DOIDN_STATIC_LIB)
endif()

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    common dnnl_common dnnl_cpu
)

message(STATUS "Setting target version ${PROJECT_VERSION}")
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION "0")

## -----------------------------------------------------------------------------
## OIDN: Examples
## -----------------------------------------------------------------------------

add_subdirectory(examples)

## -----------------------------------------------------------------------------
## OIDN: Install and packaging
## -----------------------------------------------------------------------------

include(oidn_install)

# Has to be last
include(CPack)
