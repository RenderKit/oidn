## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(API_SOURCES
  api.cpp
)

add_library(${PROJECT_NAME} ${OIDN_LIB_TYPE} ${API_SOURCES} ${OIDN_RESOURCE})
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION ${PROJECT_VERSION_MAJOR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_core)
oidn_strip_symbols(${PROJECT_NAME})