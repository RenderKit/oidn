## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(CMakeDependentOption)

add_library(${PROJECT_NAME}_utils STATIC
  arg_parser.h
  arg_parser.cpp
  device_info.h
  image_buffer.h
  image_buffer.cpp
  image_io.h
  image_io.cpp
  random.h
  timer.h
)

target_link_libraries(${PROJECT_NAME}_utils PUBLIC ${PROJECT_NAME}_common ${PROJECT_NAME})

# OpenImageIO
option(OIDN_APPS_OPENIMAGEIO "Enable OpenImageIO support in example and test apps" OFF)
if(OIDN_APPS_OPENIMAGEIO)
  find_package(OpenImageIO REQUIRED)
  if(OPENIMAGEIO_VERSION VERSION_GREATER_EQUAL "2.3")
    # OpenImageIO 2.3+ requires C++14
    set_property(TARGET ${PROJECT_NAME}_utils PROPERTY CXX_STANDARD 14)
  endif()
  target_compile_definitions(${PROJECT_NAME}_utils PRIVATE OIDN_USE_OPENIMAGEIO)
  target_link_libraries(${PROJECT_NAME}_utils PRIVATE OpenImageIO::OpenImageIO)
endif()
