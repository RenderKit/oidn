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

add_library(OpenImageDenoise_device_metal SHARED ${OIDN_METAL_SOURCES} ${OIDN_RESOURCE_FILE})

metallib_target_add_sources(OpenImageDenoise_device_metal metal_kernels
  metal_kernels.metal
  INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}
  COMPILE_OPTIONS
    -std=metal3.0 -fno-fast-math
)

set_property(TARGET OpenImageDenoise_device_metal PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET OpenImageDenoise_device_metal PROPERTY CXX_STANDARD 17)

target_compile_definitions(OpenImageDenoise_device_metal PRIVATE OIDN_COMPILE_METAL_HOST)

target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Foundation")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Metal")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShadersGraph")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShaders")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE OpenImageDenoise_core)

oidn_strip_symbols(OpenImageDenoise_device_metal)
oidn_install_module(OpenImageDenoise_device_metal)
