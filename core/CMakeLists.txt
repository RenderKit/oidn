## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(CORE_SOURCES
  autoexposure.h
  buffer.h
  buffer.cpp
  color.h
  color.cpp
  concat_conv.h
  concat_conv.cpp
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
  image_accessor.h
  image_copy.h
  image.h
  image.cpp
  input_process.h
  input_process.cpp
  math.h
  module.h
  module.cpp
  network.h
  network.cpp
  op.h
  output_process.h
  output_process.cpp
  pool.h
  pool.cpp
  progress.h
  progress.cpp
  ref.h
  reorder.h
  reorder.cpp
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
  weights.h
  weights.cpp
)

add_library(${PROJECT_NAME}_core SHARED ${CORE_SOURCES})
set_property(TARGET ${PROJECT_NAME}_core PROPERTY VERSION ${PROJECT_VERSION})

target_include_directories(${PROJECT_NAME}_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME}_core
  PUBLIC
    ${PROJECT_NAME}_common
  PRIVATE
    ${PROJECT_NAME}_weights
)

oidn_export_all_symbols(${PROJECT_NAME}_core)