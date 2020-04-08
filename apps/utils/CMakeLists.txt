## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(CMakeDependentOption)

add_library(utils STATIC
  arg_parser.h
  image_io.h
  image_io.cpp
)

target_link_libraries(utils PUBLIC common)

# OpenImageIO
find_package(OpenImageIO)
CMAKE_DEPENDENT_OPTION(OIDN_APPS_OPENIMAGEIO "Enable OpenImageIO support in example and test apps" OFF "OpenImageIO_FOUND" OFF)
if(OIDN_APPS_OPENIMAGEIO)
  target_compile_definitions(utils PRIVATE OIDN_USE_OPENIMAGEIO)
  target_link_libraries(utils PRIVATE OpenImageIO::OpenImageIO)
endif()
