## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(SYCL_SOURCES
  sycl_common.h
  sycl_conv_gen.h
  sycl_conv_gen9.cpp
  sycl_conv_xehpc.cpp
  sycl_conv_xehpg.cpp
  sycl_conv.h
  sycl_device.h
  sycl_device.cpp
  sycl_engine.h
  sycl_engine.cpp
  sycl_external_buffer.h
  sycl_external_buffer.cpp
  sycl_module.cpp
)

set_source_files_properties(${SYCL_SOURCES} PROPERTIES COMPILE_FLAGS "${OIDN_CXX_FLAGS_SYCL}")

add_library(${PROJECT_NAME}_device_sycl SHARED ${SYCL_SOURCES} ${GPU_SOURCES})
set_property(TARGET ${PROJECT_NAME}_device_sycl PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME}_device_sycl PROPERTY CXX_STANDARD 17)
target_link_options(${PROJECT_NAME}_device_sycl PRIVATE "-fsycl")
target_link_libraries(${PROJECT_NAME}_device_sycl PRIVATE ${PROJECT_NAME}_core ze_loader)
oidn_strip_symbols(${PROJECT_NAME}_device_sycl)