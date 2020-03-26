## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(CMakeDependentOption)

find_package(OpenImageIO)
CMAKE_DEPENDENT_OPTION(OIDN_EXAMPLES_OPENIMAGEIO "Enables OpenImageIO support in the examples" OFF "OpenImageIO_FOUND" OFF)

macro(add_example EXAMPLE_NAME)
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp image_io.h image_io.cpp cli.h ../common/oidn.rc)
  target_link_libraries(${EXAMPLE_NAME} PRIVATE common ${PROJECT_NAME})
  install(TARGETS ${EXAMPLE_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT examples)
  if(OIDN_EXAMPLES_OPENIMAGEIO)
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE OIDN_USE_OPENIMAGEIO)
    target_link_libraries(${EXAMPLE_NAME} PUBLIC OpenImageIO::OpenImageIO)
  endif()
endmacro()

add_example(denoise)

