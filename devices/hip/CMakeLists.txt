## Copyright 2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

# Set common paths
set(OIDN_ROOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Get the library version
include(${OIDN_ROOT_SOURCE_DIR}/cmake/oidn_version.cmake)

# Project
project(OpenImageDenoise_device_hip
  VERSION ${OIDN_VERSION}
  LANGUAGES CXX
)

# Set the CMake module path
list(APPEND CMAKE_MODULE_PATH ${OIDN_ROOT_SOURCE_DIR}/cmake)

# Common
include(oidn_common_external)

# Set the GPU targets: RDNA2, RDNA3, RDNA4
# FIXME: Older versions of the HIP runtime have a bug which may cause a crash if the kernels are
# not compiled for all targets detected in the system (it seems mostly APUs/integrated GPUs).
# As a workaround, we compile for more targets then we actually support to avoid this crash.
set(GPU_TARGETS "gfx902,gfx909,gfx90c,gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103,gfx1200,gfx1201" CACHE INTERNAL "")
set(AMDGPU_TARGETS ${GPU_TARGETS} CACHE INTERNAL "")

# Find HIP
find_package(hip REQUIRED)

set(OIDN_HIP_SOURCES
  ck_conv.h
  ck_conv_dl.cpp
  ck_conv_wmma.cpp
  hip_conv.h
  hip_conv.cpp
  hip_device.h
  hip_device.cpp
  hip_engine.h
  hip_engine.cpp
  hip_external_buffer.h
  hip_external_buffer.cpp
  hip_module.cpp
)

add_library(OpenImageDenoise_device_hip SHARED ${OIDN_HIP_SOURCES} ${OIDN_GPU_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_device_hip PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_hip
  CXX_STANDARD 17
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_hip PROPERTIES VERSION ${PROJECT_VERSION})
endif()

configure_file(
  "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/include/ck/config.h.in"
  "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/include/ck/config.h"
)

target_include_directories(OpenImageDenoise_device_hip
  PRIVATE
    "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/include"
    "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/library/include"
)

target_compile_definitions(OpenImageDenoise_device_hip PRIVATE
  CK_ENABLE_INT8
  CK_ENABLE_FP16
  DL_KERNELS
  CK_USE_WMMA
  CK_USE_OCP_FP8
)

set_source_files_properties(
  ck_conv_wmma.cpp
  PROPERTIES COMPILE_FLAGS "-mcumode -mno-wavefrontsize64" # Navi3x optimizations
)

# Fix warning: ignoring return value of function declared with 'nodiscard' attribute (HIP API)
target_compile_options(OpenImageDenoise_device_hip PRIVATE -Wno-unused-result)
if(WIN32)
  # Fix warning: __declspec attribute 'dllexport' is not supported
  target_compile_options(OpenImageDenoise_device_hip PRIVATE -Wno-ignored-attributes)
endif()

target_link_libraries(OpenImageDenoise_device_hip PRIVATE OpenImageDenoise_core hip::device)
oidn_strip_symbols(OpenImageDenoise_device_hip)
oidn_install_module(OpenImageDenoise_device_hip)