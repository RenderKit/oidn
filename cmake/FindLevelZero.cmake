## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(FindPackageHandleStandardArgs)

if(NOT LEVEL_ZERO_ROOT AND NOT $ENV{LEVEL_ZERO_ROOT} STREQUAL "")
  set(LEVEL_ZERO_ROOT "$ENV{LEVEL_ZERO_ROOT}")
endif()

find_path(LevelZero_INCLUDE_DIR
  NAMES
    level_zero/ze_api.h
  HINTS
    ${LEVEL_ZERO_ROOT}
  PATH_SUFFIXES
    include
)
mark_as_advanced(LevelZero_INCLUDE_DIR)

find_library(LevelZero_LIBRARY
  NAMES
    ze_loader
  HINTS
    ${LEVEL_ZERO_ROOT}
  PATH_SUFFIXES
    lib
    lib64
    lib/x64
)
mark_as_advanced(LevelZero_LIBRARY)

set(LevelZero_INCLUDE_DIRS ${LevelZero_INCLUDE_DIR})
set(LevelZero_LIBRARIES ${LevelZero_LIBRARY})

find_package_handle_standard_args(LevelZero
  FOUND_VAR LevelZero_FOUND
  REQUIRED_VARS LevelZero_INCLUDE_DIR LevelZero_LIBRARY
)

if(LevelZero_FOUND AND NOT TARGET LevelZero::LevelZero)
  add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
  set_target_properties(LevelZero::LevelZero PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${LevelZero_LIBRARY}"
  )
endif()