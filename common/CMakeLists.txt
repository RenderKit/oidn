## Copyright 2009-2020 Intel Corporation
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
  tensor.h
  thread.h
  thread.cpp
  timer.h
  tza.h
  tza.cpp
)

if(OIDN_STATIC_LIB)
  target_compile_definitions(common PUBLIC -DOIDN_STATIC_LIB)
endif()

target_include_directories(common
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(common PUBLIC ${CMAKE_THREAD_LIBS_INIT} TBB)
