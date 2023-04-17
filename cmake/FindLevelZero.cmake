## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.11)

include(FindPackageHandleStandardArgs)
include(FetchContent)

if(NOT LEVEL_ZERO_ROOT AND NOT $ENV{LEVEL_ZERO_ROOT} STREQUAL "")
  set(LEVEL_ZERO_ROOT "$ENV{LEVEL_ZERO_ROOT}")
endif()

find_path(LevelZero_INCLUDE_DIR
  NAMES
    level_zero/ze_api.h
  HINTS
    ${LEVEL_ZERO_ROOT}
    ENV CPATH
  PATH_SUFFIXES
    include
    sycl # might ship with DPC++ compiler
)
mark_as_advanced(LevelZero_INCLUDE_DIR)

find_library(LevelZero_LIBRARY
  NAMES
    ze_loader
  HINTS
    ${LEVEL_ZERO_ROOT}
    ENV LIBRARY_PATH
  PATH_SUFFIXES
    lib
    lib64
    lib/x64
)
mark_as_advanced(LevelZero_LIBRARY)

if(NOT LevelZero_INCLUDE_DIR OR (WIN32 AND NOT LevelZero_LIBRARY))
  message(STATUS "oneAPI Level Zero SDK not found, will be downloaded")

  # Download the Level Zero SDK for Windows, which includes the headers too
  FetchContent_Declare(level-zero-sdk
    URL https://github.com/oneapi-src/level-zero/releases/download/v1.8.8/level-zero_1.8.8_win-sdk.zip
    URL_HASH SHA256=2dde077823e612e7ab33034faa86323fadc0147789412ad6f9f804de0fd783e8
  )
  FetchContent_MakeAvailable(level-zero-sdk)

  set(LevelZero_INCLUDE_DIR ${level-zero-sdk_SOURCE_DIR}/include)
  if(WIN32)
    set(LevelZero_LIBRARY ${level-zero-sdk_SOURCE_DIR}/lib/ze_loader.lib)
  endif()
endif()

if(NOT WIN32 AND NOT LevelZero_LIBRARY)
  message(STATUS "oneAPI Level Zero Loader not found, will be downloaded and built from source")

  # We need some workarounds for building the Level Zero Loader
  set(CMAKE_CXX_FLAGS_BAK "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")

  # Download the Level Zero Loader source and include it in the build
  FetchContent_Declare(level-zero-loader
    GIT_REPOSITORY https://github.com/oneapi-src/level-zero.git
    GIT_TAG        32c4431d731bc2ba7b5b88b32335063efa65e076     # v1.8.8
  )
  FetchContent_MakeAvailable(level-zero-loader)

  # Restore original compile flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BAK}")

  set(LevelZero_LIBRARY ze_loader) # name of the target we will build

  find_package_handle_standard_args(LevelZero
    FOUND_VAR LevelZero_FOUND
    REQUIRED_VARS LevelZero_INCLUDE_DIR LevelZero_LIBRARY
  )

  if(LevelZero_FOUND AND NOT TARGET LevelZero::LevelZero)
    add_library(LevelZero::LevelZero INTERFACE IMPORTED)
    set_target_properties(LevelZero::LevelZero PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${LevelZero_LIBRARY}"
    )
  endif()
else()
  find_package_handle_standard_args(LevelZero
    FOUND_VAR LevelZero_FOUND
    REQUIRED_VARS LevelZero_INCLUDE_DIR LevelZero_LIBRARY
  )

  if(LevelZero_FOUND AND NOT TARGET LevelZero::LevelZero)
    add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
    set_target_properties(LevelZero::LevelZero PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIR}"
      IMPORTED_LOCATION "${LevelZero_LIBRARY}"
    )
  endif()
endif()

if(LevelZero_FOUND)
  set(LevelZero_INCLUDE_DIRS ${LevelZero_INCLUDE_DIR})
  set(LevelZero_LIBRARIES ${LevelZero_LIBRARY})
endif()