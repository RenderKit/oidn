## Copyright 2023 Apple Inc.
## Copyright 2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.21)

include(oidn_metal)

set(OIDN_METAL_SOURCES
  metal_buffer.mm
  metal_buffer.h
  metal_common.h
  metal_common.mm
  metal_device.mm
  metal_device.h
  metal_conv.mm
  metal_conv.h
  metal_engine.mm
  metal_engine.h
  metal_heap.h
  metal_heap.mm
  metal_module.mm
)

add_library(OpenImageDenoise_device_metal ${OIDN_LIB_TYPE} ${OIDN_METAL_SOURCES} ${OIDN_RESOURCE_FILE})

metallib_target_add_sources(OpenImageDenoise_device_metal metal_kernels
  metal_kernels.metal
  INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}
  COMPILE_OPTIONS
    -std=metal3.0 -fno-fast-math -m${OIDN_APPLE_SDK}-version-min=${OIDN_APPLE_SDK_VERSION_MIN}
)

set_target_properties(OpenImageDenoise_device_metal PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_metal
  CXX_STANDARD 17
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_metal PROPERTIES VERSION ${PROJECT_VERSION})
endif()

target_compile_definitions(OpenImageDenoise_device_metal PRIVATE OIDN_COMPILE_METAL_HOST)

target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Foundation")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Metal")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShadersGraph")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShaders")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE OpenImageDenoise_core)

if(OIDN_STATIC_LIB)
  oidn_install_static_module(OpenImageDenoise_device_metal)
  target_link_libraries(OpenImageDenoise PRIVATE OpenImageDenoise_device_metal)
else()
  oidn_strip_symbols(OpenImageDenoise_device_metal)
  oidn_install_module(OpenImageDenoise_device_metal)
endif()
