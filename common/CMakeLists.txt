## Copyright 2009-2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

add_library(${PROJECT_NAME}_common STATIC
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
)

if(OIDN_STATIC_LIB)
  install(TARGETS {PROJECT_NAME}_common
    EXPORT
      ${PROJECT_NAME}_Export
    ARCHIVE
      DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  )
endif()
