## Copyright 2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

add_subdirectory(utils)

macro(oidn_add_app APP_NAME)
  add_executable(${APP_NAME} ${ARGN} ${OIDN_RESOURCE_FILE})
  target_link_libraries(${APP_NAME} PRIVATE OpenImageDenoise_common OpenImageDenoise_utils OpenImageDenoise)
  install(TARGETS ${APP_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT apps)
endmacro()

oidn_add_app(oidnDenoise oidnDenoise.cpp)
oidn_add_app(oidnBenchmark oidnBenchmark.cpp)
oidn_add_app(oidnTest oidnTest.cpp "${PROJECT_SOURCE_DIR}/external/catch.hpp")