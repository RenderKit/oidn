## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

macro(add_example EXAMPLE_NAME)
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp image_io.h image_io.cpp cli.h ../common/oidn.rc)
  target_link_libraries(${EXAMPLE_NAME} PRIVATE common ${PROJECT_NAME})
  install(TARGETS ${EXAMPLE_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT examples)
endmacro()

add_example(denoise)

