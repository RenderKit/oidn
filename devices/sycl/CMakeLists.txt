## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(LevelZero REQUIRED)

set(SYCL_SOURCES
  sycl_common.h
  sycl_conv_gen.h
  sycl_conv_gen9.cpp
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

if(UNIX)
  set(SYCL_SOURCES ${SYCL_SOURCES} sycl_conv_xehpc.cpp)
endif()

set_source_files_properties(${SYCL_SOURCES} PROPERTIES COMPILE_FLAGS "${OIDN_CXX_FLAGS_SYCL}")

add_library(${PROJECT_NAME}_device_sycl SHARED ${SYCL_SOURCES} ${GPU_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME}_device_sycl PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME}_device_sycl PROPERTY CXX_STANDARD 17)
target_link_options(${PROJECT_NAME}_device_sycl PRIVATE "-fsycl")
target_link_libraries(${PROJECT_NAME}_device_sycl PRIVATE ${PROJECT_NAME}_core LevelZero::LevelZero)
oidn_strip_symbols(${PROJECT_NAME}_device_sycl)
oidn_install_module(${PROJECT_NAME}_device_sycl)