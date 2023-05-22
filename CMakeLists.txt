## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

# Get the library version
include(cmake/oidn_version.cmake)

# Main project
project(OpenImageDenoise
  VERSION ${OIDN_VERSION}
  LANGUAGES C CXX
)

# Set common paths
set(OIDN_ROOT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(OIDN_ROOT_BINARY_DIR ${PROJECT_BINARY_DIR})
set(OIDN_INSTALL_RPATH_PREFIX ${CMAKE_INSTALL_PREFIX})

# Set the CMake module path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Common
include(oidn_common)

# Devices
option(OIDN_DEVICE_CPU "Enable CPU device." ON)
if(NOT APPLE)
  if(OIDN_ARCH STREQUAL "X64")
    option(OIDN_DEVICE_SYCL "Enable SYCL device." OFF)
  endif()
  option(OIDN_DEVICE_CUDA "Enable CUDA device." OFF)
  option(OIDN_DEVICE_HIP "Enable HIP device." OFF)
endif()

# Filters
option(OIDN_FILTER_RT "Include trained weights of the RT filter." ON)
option(OIDN_FILTER_RTLIGHTMAP "Include trained weights of the RTLightmap filter." ON)

# Install
option(OIDN_INSTALL_DEPENDENCIES "Install Open Image Denoise dependencies" OFF)

## -----------------------------------------------------------------------------
## Weights
## -----------------------------------------------------------------------------

set(OIDN_WEIGHTS)

if(OIDN_FILTER_RT)
  list(APPEND OIDN_WEIGHTS
    weights/rt_alb.tza
    weights/rt_hdr.tza
    weights/rt_hdr_alb.tza
    weights/rt_hdr_alb_nrm.tza
    weights/rt_hdr_calb_cnrm.tza
    weights/rt_ldr.tza
    weights/rt_ldr_alb.tza
    weights/rt_ldr_alb_nrm.tza
    weights/rt_ldr_calb_cnrm.tza
    weights/rt_nrm.tza
  )
endif()

if(OIDN_FILTER_RTLIGHTMAP)
  list(APPEND OIDN_WEIGHTS
    weights/rtlightmap_hdr.tza
    weights/rtlightmap_dir.tza
  )
endif()

oidn_generate_cpp_from_blob(OIDN_WEIGHTS_SOURCES "${OIDN_NAMESPACE}::blobs::weights" ${OIDN_WEIGHTS})
if(OIDN_WEIGHTS_SOURCES)
  add_library(OpenImageDenoise_weights OBJECT ${OIDN_WEIGHTS_SOURCES})
endif()

## -----------------------------------------------------------------------------
## Libraries
## -----------------------------------------------------------------------------

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(api)
add_subdirectory(devices)

# Export targets from the build tree for external projects
export(EXPORT OpenImageDenoise_Exports
  FILE ${OIDN_BUILD_TREE_EXPORT_FILE}
)

## -----------------------------------------------------------------------------
## Apps
## -----------------------------------------------------------------------------

option(OIDN_APPS "Enable building example and test apps." ON)
if(OIDN_APPS)
  add_subdirectory(apps)
endif()

## -----------------------------------------------------------------------------
## Install and packaging
## -----------------------------------------------------------------------------

# Install headers
install(DIRECTORY include/OpenImageDenoise
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT devel
  PATTERN "*.in" EXCLUDE
)

# Install documentation
install(
  FILES
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/readme.pdf
    ${PROJECT_SOURCE_DIR}/CHANGELOG.md
    ${PROJECT_SOURCE_DIR}/LICENSE.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs-DPCPP.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs-oneDNN.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs-oneTBB.txt
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT lib
)

# Install generic dependencies
if(OIDN_INSTALL_DEPENDENCIES AND CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  get_filename_component(_icx_compiler_dir ${CMAKE_CXX_COMPILER} PATH)

  if(WIN32)
    file(GLOB _icx_deps LIST_DIRECTORIES FALSE
      "${_icx_compiler_dir}/../redist/intel64_win/compiler/libmmd.dll"
      "${_icx_compiler_dir}/../redist/intel64_win/compiler/svml_dispmd.dll"
    )
  else()
    file(GLOB _icx_deps LIST_DIRECTORIES FALSE
      "${_icx_compiler_dir}/../compiler/lib/intel64_lin/libsvml.so"
      "${_icx_compiler_dir}/../compiler/lib/intel64_lin/libirng.so"
      "${_icx_compiler_dir}/../compiler/lib/intel64_lin/libimf.so"
      "${_icx_compiler_dir}/../compiler/lib/intel64_lin/libintlc.so.?"
    )
  endif()

  oidn_install_lib_files(${_icx_deps})
endif()

# Install CMake configuration files
install(EXPORT OpenImageDenoise_Exports
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenImageDenoise-${OIDN_VERSION}
  FILE OpenImageDenoiseConfig.cmake
  COMPONENT devel
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(OpenImageDenoiseConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_BINARY_DIR}/OpenImageDenoiseConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenImageDenoise-${OIDN_VERSION}
  COMPONENT devel
)

# Has to be last
include(CPack)
