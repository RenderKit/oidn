## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(CMakeDependentOption)

add_library(utils STATIC
  arg_parser.h
  image_io.h
  image_io.cpp
  random.h
)

target_link_libraries(utils PUBLIC common ${PROJECT_NAME})

# OpenImageIO
option(OIDN_APPS_OPENIMAGEIO "Enable OpenImageIO support in example and test apps" OFF)
if(OIDN_APPS_OPENIMAGEIO)
  find_package(OpenImageIO REQUIRED)
  if(OPENIMAGEIO_VERSION VERSION_GREATER_EQUAL "2.3")
    # OpenImageIO 2.3+ requires C++14
    set_property(TARGET utils PROPERTY CXX_STANDARD 14)
  endif()
  target_compile_definitions(utils PRIVATE OIDN_USE_OPENIMAGEIO)
  target_link_libraries(utils PRIVATE OpenImageIO::OpenImageIO)
endif()
