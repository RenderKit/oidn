## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Generate config.h
configure_file(
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h.in"
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h"
)

# Generate list of symbols to export
configure_file(
  "${PROJECT_SOURCE_DIR}/common/export.linux.map.in"
  "${PROJECT_SOURCE_DIR}/common/export.linux.map"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/common/export.macos.map.in"
  "${PROJECT_SOURCE_DIR}/common/export.macos.map"
)

add_library(${PROJECT_NAME}_common OBJECT
  ${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h
  ${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/oidn.h
  ${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/oidn.hpp
  common.h
  common.cpp
  half.h
  half.cpp
  oidn_utils.h
  oidn_utils.cpp
  platform.h
  platform.cpp
)

target_include_directories(${PROJECT_NAME}_common
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

install(TARGETS ${PROJECT_NAME}_common EXPORT ${PROJECT_NAME}_Exports)