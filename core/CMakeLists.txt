## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

set(OIDN_CORE_SOURCES
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
  generic_graph.h
  generic_graph.cpp
  graph.h
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
  output_process.h
  output_process.cpp
  pool.h
  pool.cpp
  progress.h
  progress.cpp
  ref.h
  rt_filter.h
  rt_filter.cpp
  rtlightmap_filter.h
  rtlightmap_filter.cpp
  scratch.h
  scratch.cpp
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

add_library(OpenImageDenoise_core SHARED ${OIDN_CORE_SOURCES} ${OIDN_RESOURCE_FILE})
set_property(TARGET OpenImageDenoise_core PROPERTY VERSION ${PROJECT_VERSION})

target_link_libraries(OpenImageDenoise_core
  PUBLIC
    OpenImageDenoise_common
  PRIVATE
    OpenImageDenoise_weights Threads::Threads ${CMAKE_DL_LIBS}
)

oidn_export_all_symbols(OpenImageDenoise_core)

install(TARGETS OpenImageDenoise_core
  EXPORT
    OpenImageDenoise_Exports
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
    NAMELINK_SKIP
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
)