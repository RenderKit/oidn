## Copyright 2023 Apple Inc.
## SPDX-License-Identifier: Apache-2.0

find_library(METAL_LIBRARY Metal)
find_library(METALKIT_LIBRARY MetalKit)

# Shaders
set(SRC_METAL_KERNEL
  metal_input_process_kernel.metal
  metal_output_process_kernel.metal
  metal_autoexposure_kernel.metal
  metal_kernel_common.metal
)
set_source_files_properties(${SRC_METAL_KERNEL} PROPERTIES LANGUAGE METAL)

set(OIDN_METAL_SOURCES
  metal_module.mm
  metal_common.h
  metal_common.mm
  metal_engine.mm
  metal_engine.h
  metal_device.mm
  metal_device.h
  metal_graph.mm
  metal_graph.h
  metal_buffer.mm
  metal_buffer.h
  metal_autoexposure.h
  metal_autoexposure.mm
  metal_image_copy.h
  metal_image_copy.mm
  metal_input_process.h
  metal_input_process.mm
  metal_kernel_common.h
  metal_kernel_constants.h
  metal_transfer_function.h
  metal_output_process.h
  metal_output_process.mm
  ${SRC_METAL_KERNEL}
)

# Set metal version
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=macos-metal2.0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=macos-metal2.0")

add_library(OpenImageDenoise_device_metal SHARED ${OIDN_METAL_SOURCES} ${OIDN_RESOURCE_FILE})
set_property(TARGET OpenImageDenoise_device_metal PROPERTY VERSION ${PROJECT_VERSION})

target_compile_definitions(OpenImageDenoise_device_metal PRIVATE OIDN_METAL)
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Foundation")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework Metal")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShadersGraph")
target_link_libraries(OpenImageDenoise_device_metal PRIVATE "-framework MetalPerformanceShaders")

target_link_libraries(OpenImageDenoise_device_metal PRIVATE OpenImageDenoise_core)
oidn_strip_symbols(OpenImageDenoise_device_metal)
oidn_install_module(OpenImageDenoise_device_metal)
