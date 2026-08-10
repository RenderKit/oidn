## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# These files used to be generated into the source tree, where build directories with different
# options would overwrite each other's copies. Delete any leftovers, which would take precedence
# over the generated ones because they are next to the headers which include them.
file(REMOVE
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h"
  "${PROJECT_SOURCE_DIR}/common/export.linux.map"
  "${PROJECT_SOURCE_DIR}/common/export.macos.map"
)

# Generate config.h
configure_file(
  "${PROJECT_SOURCE_DIR}/include/OpenImageDenoise/config.h.in"
  "${PROJECT_BINARY_DIR}/include/OpenImageDenoise/config.h"
)

# Generate list of symbols to export
configure_file(
  "${PROJECT_SOURCE_DIR}/common/export.linux.map.in"
  "${PROJECT_BINARY_DIR}/common/export.linux.map"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/common/export.macos.map.in"
  "${PROJECT_BINARY_DIR}/common/export.macos.map"
)

add_library(OpenImageDenoise_common OBJECT
  ${PROJECT_BINARY_DIR}/include/OpenImageDenoise/config.h
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
  timer.h
)

set_target_properties(OpenImageDenoise_common PROPERTIES OUTPUT_NAME ${OIDN_LIBRARY_NAME}_common)

target_include_directories(OpenImageDenoise_common
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/OpenImageDenoise> # config.h
)

install(TARGETS OpenImageDenoise_common EXPORT OpenImageDenoise_Exports)
