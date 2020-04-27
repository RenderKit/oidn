## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

macro(oidn_add_app APP_NAME)
  add_executable(${APP_NAME} ${ARGN} ${OIDN_RESOURCE})
  target_link_libraries(${APP_NAME} PRIVATE common utils ${PROJECT_NAME})
  install(TARGETS ${APP_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT apps)
endmacro()

add_subdirectory(utils)
add_subdirectory(oidnDenoise)
add_subdirectory(oidnBench)
add_subdirectory(oidnTest)