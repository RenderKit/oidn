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

# Set the GPU targets: RDNA2, RDNA3, RDNA4, and CDNA4 (gfx950) when the compiler
# accepts it. gfx950 cannot share a translation unit with the RDNA WMMA/DL
# sources: those kernels are wave32 and use RDNA matrix instructions, while the
# CDNA MFMA kernels are wave64. The conv sources below opt out per file.
# FIXME: Older versions of the HIP runtime have a bug which may cause a crash if the kernels are
# not compiled for all targets detected in the system (it seems mostly APUs/integrated GPUs).
# As a workaround, we compile for more targets then we actually support to avoid this crash.
set(OIDN_HIP_RDNA_TARGETS "gfx902,gfx909,gfx90c,gfx1030,gfx1031,gfx1032,gfx1033,gfx1034,gfx1035,gfx1036,gfx1100,gfx1101,gfx1102,gfx1103,gfx1150,gfx1151,gfx1152,gfx1200,gfx1201")
option(OIDN_HIP_GFX950 "Build HIP kernels for CDNA4 gfx950 (Instinct MI350)." ON)
if(NOT OIDN_HIP_TARGETS)
  set(OIDN_HIP_TARGETS "${OIDN_HIP_RDNA_TARGETS}")
  if(OIDN_HIP_GFX950)
    set(_oidn_gfx950_probe "${CMAKE_CURRENT_BINARY_DIR}/oidn_gfx950_probe.hip")
    file(WRITE "${_oidn_gfx950_probe}" "__global__ void oidn_gfx950_probe() {}\n")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" -x hip --offload-arch=gfx950 --cuda-device-only
              -c "${_oidn_gfx950_probe}" -o "${CMAKE_CURRENT_BINARY_DIR}/oidn_gfx950_probe.o"
      RESULT_VARIABLE _oidn_gfx950_probe_result
      OUTPUT_QUIET ERROR_QUIET
    )
    if(_oidn_gfx950_probe_result EQUAL 0)
      string(APPEND OIDN_HIP_TARGETS ",gfx950")
    else()
      message(STATUS "HIP compiler does not accept gfx950; CDNA4 kernels will not be built")
    endif()
  endif()
endif()
set(GPU_TARGETS "${OIDN_HIP_TARGETS}" CACHE INTERNAL "")
set(AMDGPU_TARGETS ${GPU_TARGETS} CACHE INTERNAL "")
if(OIDN_HIP_TARGETS MATCHES "(^|,)gfx950(,|$)")
  set(OIDN_HIP_GFX950_ENABLED ON)
else()
  set(OIDN_HIP_GFX950_ENABLED OFF)
endif()

# Enable parallel Clang jobs per source file: Clang generates code for each GPU
# ISA target in parallel, which significantly speeds up HIP compilation.
# Defaults to the number of logical CPU cores; set to 1 to disable.
if(NOT DEFINED HIP_CLANG_NUM_PARALLEL_JOBS)
  include(ProcessorCount)
  ProcessorCount(HIP_CLANG_NUM_PARALLEL_JOBS)
  if(HIP_CLANG_NUM_PARALLEL_JOBS EQUAL 0)
    set(HIP_CLANG_NUM_PARALLEL_JOBS 1)
  endif()
endif()

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
  hip_external_semaphore.h
  hip_external_semaphore.cpp
  hip_module.cpp
)
if(OIDN_HIP_GFX950_ENABLED)
  list(APPEND OIDN_HIP_SOURCES ck_conv_mfma.cpp)
endif()

add_library(OpenImageDenoise_device_hip SHARED ${OIDN_HIP_SOURCES} ${OIDN_GPU_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_device_hip PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_hip
  CXX_STANDARD 17
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_hip PROPERTIES VERSION ${PROJECT_VERSION})
endif()

# Generate CK's config header in the build tree. The pinned submodule already
# contains a config.h from a previous build, and writing into the submodule
# dirties it.
set(_oidn_ck_config_dir "${CMAKE_CURRENT_BINARY_DIR}/ck_config")
configure_file(
  "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/include/ck/config.h.in"
  "${_oidn_ck_config_dir}/ck/config.h"
)

target_include_directories(OpenImageDenoise_device_hip
  PRIVATE
    "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/include"
    "${OIDN_ROOT_SOURCE_DIR}/external/composable_kernel/library/include"
)
target_include_directories(OpenImageDenoise_device_hip
  BEFORE PRIVATE
    "${_oidn_ck_config_dir}"
)

target_compile_definitions(OpenImageDenoise_device_hip PRIVATE
  CK_ENABLE_INT8
  CK_ENABLE_FP16
  CK_USE_OCP_FP8
)
# DL/WMMA switches select RDNA kernels inside Composable Kernel. Keep them off
# the gfx950 MFMA translation unit, which uses the XDL implementation instead.
set_source_files_properties(ck_conv_dl.cpp ck_conv_wmma.cpp PROPERTIES
  COMPILE_DEFINITIONS "DL_KERNELS;CK_USE_WMMA"
)
if(OIDN_HIP_GFX950_ENABLED)
  target_compile_definitions(OpenImageDenoise_device_hip PRIVATE OIDN_HIP_GFX950)
endif()

# WMMA kernels are wave32. Drop gfx950 from that translation unit, and compile
# the MFMA kernels for gfx950 only. --no-offload-arch is applied per source so
# it comes after the hip::device --offload-arch list.
string(REPLACE "," ";" _oidn_hip_target_list "${OIDN_HIP_TARGETS}")
set(_oidn_wmma_flags -mcumode -mno-wavefrontsize64)
set(_oidn_dl_flags)
set(_oidn_mfma_flags -mwavefrontsize64)
foreach(_oidn_hip_target IN LISTS _oidn_hip_target_list)
  if(_oidn_hip_target STREQUAL "gfx950")
    list(APPEND _oidn_wmma_flags "--no-offload-arch=gfx950")
    list(APPEND _oidn_dl_flags "--no-offload-arch=gfx950")
  else()
    list(APPEND _oidn_mfma_flags "--no-offload-arch=${_oidn_hip_target}")
  endif()
endforeach()

set_source_files_properties(ck_conv_wmma.cpp PROPERTIES COMPILE_OPTIONS "${_oidn_wmma_flags}")
if(_oidn_dl_flags)
  set_source_files_properties(ck_conv_dl.cpp PROPERTIES COMPILE_OPTIONS "${_oidn_dl_flags}")
endif()
if(OIDN_HIP_GFX950_ENABLED)
  set_source_files_properties(ck_conv_mfma.cpp PROPERTIES COMPILE_OPTIONS "${_oidn_mfma_flags}")
endif()

# Fix error with ROCm 7: constexpr function never produces a constant expression
target_compile_options(OpenImageDenoise_device_hip PRIVATE -Wno-invalid-constexpr)
# Fix warning: ignoring return value of function declared with 'nodiscard' attribute (HIP API)
target_compile_options(OpenImageDenoise_device_hip PRIVATE -Wno-unused-result)
if(WIN32)
  # Fix warning: __declspec attribute 'dllexport' is not supported
  target_compile_options(OpenImageDenoise_device_hip PRIVATE -Wno-ignored-attributes)
endif()

target_link_libraries(OpenImageDenoise_device_hip PRIVATE OpenImageDenoise_core hip::device)
oidn_strip_symbols(OpenImageDenoise_device_hip)
oidn_install_module(OpenImageDenoise_device_hip)