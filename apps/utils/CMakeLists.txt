## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(CMakeDependentOption)

set(OIDN_UTILS_SOURCES
  arg_parser.h
  arg_parser.cpp
  device_info.h
  image_buffer.h
  image_buffer.cpp
  image_io.h
  image_io.cpp
  random.h
)

if(NOT OIDN_API_NAMESPACE)
  list(APPEND OIDN_UTILS_SOURCES dummy.c)
endif()

add_library(OpenImageDenoise_utils STATIC ${OIDN_UTILS_SOURCES})
set_target_properties(OpenImageDenoise_utils PROPERTIES OUTPUT_NAME ${OIDN_LIBRARY_NAME}_utils)
target_link_libraries(OpenImageDenoise_utils PUBLIC OpenImageDenoise_common OpenImageDenoise)

# OpenImageIO
option(OIDN_APPS_OPENIMAGEIO "Enable OpenImageIO support in example and test apps." OFF)
if(OIDN_APPS_OPENIMAGEIO)
  find_package(OpenImageIO REQUIRED)
  if(OPENIMAGEIO_VERSION VERSION_GREATER_EQUAL "3")
    # OpenImageIO 3.x requires C++17
    set_target_properties(OpenImageDenoise_utils PROPERTIES CXX_STANDARD 17)
  elseif(OPENIMAGEIO_VERSION VERSION_GREATER_EQUAL "2.3")
    # OpenImageIO 2.3+ requires C++14
    set_target_properties(OpenImageDenoise_utils PROPERTIES CXX_STANDARD 14)
  endif()
  target_compile_definitions(OpenImageDenoise_utils PRIVATE OIDN_USE_OPENIMAGEIO)
  target_link_libraries(OpenImageDenoise_utils PRIVATE OpenImageIO::OpenImageIO)
endif()
