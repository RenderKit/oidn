## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(LevelZero REQUIRED)

set(OIDN_SYCL_SOURCES
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
  list(APPEND OIDN_SYCL_SOURCES sycl_conv_xehpc.cpp)
endif()

set(OIDN_SYCL_COMPILE_FLAGS "-fsycl")
# Silence false positive "loop not unrolled" warnings
append(OIDN_SYCL_COMPILE_FLAGS "-Wno-pass-failed")
# FIXME: DPCPP issues a warning when WINAPI is used:
# warning: '__stdcall' calling convention is not supported for this target
if(WIN32)
  append(OIDN_SYCL_COMPILE_FLAGS "-Wno-ignored-attributes")
endif()

set(OIDN_SYCL_LINK_FLAGS "-fsycl")

set_source_files_properties(${OIDN_SYCL_SOURCES} PROPERTIES COMPILE_FLAGS "${OIDN_SYCL_COMPILE_FLAGS}")

add_library(${PROJECT_NAME}_device_sycl SHARED ${OIDN_SYCL_SOURCES} ${OIDN_GPU_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME}_device_sycl PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME}_device_sycl PROPERTY CXX_STANDARD 17)
target_link_options(${PROJECT_NAME}_device_sycl PRIVATE "${OIDN_SYCL_LINK_FLAGS}")
target_link_libraries(${PROJECT_NAME}_device_sycl PRIVATE ${PROJECT_NAME}_core LevelZero::LevelZero)
oidn_strip_symbols(${PROJECT_NAME}_device_sycl)
oidn_install_module(${PROJECT_NAME}_device_sycl)