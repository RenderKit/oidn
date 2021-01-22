## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

add_library(common STATIC
  barrier.h
  exception.h
  math.h
  platform.h
  platform.cpp
  ref.h
  tasking.h
  tasking.cpp
  thread.h
  thread.cpp
  timer.h
)

target_include_directories(common
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

target_link_libraries(common
  PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    TBB
)

if(OIDN_STATIC_LIB)
  install(TARGETS common
    EXPORT
      ${PROJECT_NAME}_Export
    ARCHIVE
      DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  )
endif()
