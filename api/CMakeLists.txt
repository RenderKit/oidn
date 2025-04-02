## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(API_SOURCES
  api.cpp
)

add_library(OpenImageDenoise ${OIDN_LIB_TYPE} ${API_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
endif()

target_include_directories(OpenImageDenoise
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(OpenImageDenoise PRIVATE OpenImageDenoise_core)

if(NOT OIDN_STATIC_LIB)
  oidn_strip_symbols(OpenImageDenoise)
endif()

install(TARGETS OpenImageDenoise
  EXPORT
    OpenImageDenoise_Exports
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
)