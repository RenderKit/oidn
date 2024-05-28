## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Policy: find_package() uses <PackageName>_ROOT variables
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Set build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OIDN_ROOT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OIDN_ROOT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OIDN_ROOT_BINARY_DIR})

# CMake macros
include(oidn_macros)

# Configuration types
set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
if(win32)
  if(NOT OIDN_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET)
    set(CMAKE_CONFIGURATION_TYPES "${CONFIGURATION_TYPES}"
        CACHE STRING "List of generated configurations." FORCE)
    set(OOIDN_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET ON
        CACHE INTERNAL "Default CMake configuration types set.")
  endif()
else()
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CONFIGURATION_TYPES})
endif()

# Build as shared or static library
option(OIDN_STATIC_LIB "Build Open Image Denoise as a static or hybrid static/shared library.")
mark_as_advanced(CLEAR OIDN_STATIC_LIB)
if(OIDN_STATIC_LIB)
  set(OIDN_LIB_TYPE STATIC)
else()
  set(OIDN_LIB_TYPE SHARED)
endif()

# Library name
set(OIDN_LIBRARY_NAME "OpenImageDenoise" CACHE STRING "Base name of the Open Image Denoise library files.")

# API namespace
set(OIDN_API_NAMESPACE "" CACHE STRING "C++ namespace to put API symbols into.")
if(OIDN_API_NAMESPACE)
  set(OIDN_NAMESPACE ${OIDN_API_NAMESPACE}::oidn)
else()
  set(OIDN_NAMESPACE oidn)
endif()

# File containing targets exported from the build tree for external projects
set(OIDN_BUILD_TREE_EXPORT_FILE ${OIDN_ROOT_BINARY_DIR}/cmake/oidn_targets.cmake)

# Common resource file
set(OIDN_RESOURCE_FILE ${OIDN_ROOT_SOURCE_DIR}/common/oidn.rc)

# Platform-specific settings
include(oidn_platform)

# Packaging
include(oidn_package)