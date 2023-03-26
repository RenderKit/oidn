## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

set(CORE_SOURCES
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

add_library(${PROJECT_NAME}_core SHARED ${CORE_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME}_core PROPERTY VERSION ${PROJECT_VERSION})

target_link_libraries(${PROJECT_NAME}_core
  PUBLIC
    ${PROJECT_NAME}_common
  PRIVATE
    ${PROJECT_NAME}_weights Threads::Threads ${CMAKE_DL_LIBS}
)

oidn_export_all_symbols(${PROJECT_NAME}_core)

install(TARGETS ${PROJECT_NAME}_core
  EXPORT
    ${PROJECT_NAME}_Export
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
    NAMELINK_SKIP
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
)