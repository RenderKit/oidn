## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

set(OIDN_CORE_SOURCES
  arena.h
  arena.cpp
  arena_planner.h
  arena_planner.cpp
  autoexposure.h
  buffer.h
  buffer.cpp
  color.h
  color.cpp
  concat_conv.h
  concat_conv.cpp
  concat_conv_chw.h
  concat_conv_chw.cpp
  concat_conv_hwc.h
  concat_conv_hwc.cpp
  context.h
  context.cpp
  conv.cpp
  conv.h
  conv.cpp
  data.h
  device_factory.h
  device.h
  device.cpp
  engine.h
  engine.cpp
  exception.h
  exception.cpp
  filter.h
  filter.cpp
  graph.h
  graph.cpp
  heap.h
  heap.cpp
  image_accessor.h
  image_copy.h
  image.h
  image.cpp
  input_process.h
  input_process.cpp
  math.h
  module.h
  module.cpp
  op.h
  op.cpp
  output_process.h
  output_process.cpp
  pool.h
  pool.cpp
  progress.h
  progress.cpp
  record.h
  ref.h
  rt_filter.h
  rt_filter.cpp
  rtlightmap_filter.h
  rtlightmap_filter.cpp
  subdevice.h
  subdevice.cpp
  tensor.h
  tensor.cpp
  tensor_accessor.h
  tensor_layout.h
  tensor_reorder.h
  tensor_reorder.cpp
  thread.h
  thread.cpp
  tile.h
  tza.h
  tza.cpp
  unet_filter.h
  unet_filter.cpp
  upsample.h
  upsample.cpp
  vec.h
)

# Fully static build is supported only for the CPU device
if(OIDN_STATIC_LIB AND NOT OIDN_DEVICE_SYCL AND NOT OIDN_DEVICE_CUDA AND NOT OIDN_DEVICE_HIP)
  set(OIDN_CORE_LIB_TYPE STATIC)
else()
  set(OIDN_CORE_LIB_TYPE SHARED)
endif()

add_library(OpenImageDenoise_core ${OIDN_CORE_LIB_TYPE} ${OIDN_CORE_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_core PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_core
)
target_compile_definitions(OpenImageDenoise_core PRIVATE OIDN_LIBRARY_NAME="${OIDN_LIBRARY_NAME}")

if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_core PROPERTIES VERSION ${PROJECT_VERSION})
  target_compile_definitions(OpenImageDenoise_core PRIVATE OIDN_LIBRARY_VERSIONED)
endif()

target_link_libraries(OpenImageDenoise_core
  PUBLIC
    OpenImageDenoise_common
  PRIVATE
    Threads::Threads ${CMAKE_DL_LIBS}
)

if(TARGET OpenImageDenoise_weights)
  target_link_libraries(OpenImageDenoise_core PRIVATE OpenImageDenoise_weights)
endif()

if(OIDN_CORE_LIB_TYPE STREQUAL "SHARED")
  oidn_export_all_symbols(OpenImageDenoise_core)
endif()

install(TARGETS OpenImageDenoise_core
  EXPORT
    OpenImageDenoise_Exports
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
    NAMELINK_SKIP
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
)