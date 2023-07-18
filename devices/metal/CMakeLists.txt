## Copyright 2023 Apple Inc.
## Copyright 2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_library(METAL_LIBRARY Metal)
find_library(METALKIT_LIBRARY MetalKit)

# Shaders
set(OIDN_METAL_SOURCES
  metal_buffer.mm
  metal_buffer.h
  metal_common.h
  metal_common.mm
  metal_device.mm
  metal_device.h
  metal_engine.mm
  metal_engine.h
  metal_graph.mm
  metal_graph.h
  metal_module.mm
)

set(OIDN_METAL_KERNEL_SOURCES
  metal_kernels.metal
)

set_source_files_properties(${OIDN_METAL_KERNEL_SOURCES} PROPERTIES LANGUAGE METAL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=metal3.0 -fno-fast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=metal3.0 -fno-fast-math")

add_library(OpenImageDenoise_device_metal SHARED ${OIDN_METAL_SOURCES} ${OIDN_METAL_KERNEL_SOURCES} ${OIDN_RESOURCE_FILE})
set_property(TARGET OpenImageDenoise_device_metal PROPERTY VERSION ${PROJECT_VERSION})

target_compile_definitions(OpenImageDenoise_device_metal PRIVATE OIDN_COMPILE_METAL_HOST)
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Foundation")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Metal")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShadersGraph")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShaders")

target_link_libraries(OpenImageDenoise_device_metal PRIVATE OpenImageDenoise_core)
oidn_strip_symbols(OpenImageDenoise_device_metal)
oidn_install_module(OpenImageDenoise_device_metal)
