## ======================================================================== ##
## Copyright 2009-2019 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

cmake_minimum_required(VERSION 3.0)

project(OpenImageDenoise VERSION 0.6.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

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

include(GNUInstallDirs)

# Output paths
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Configuration
option(OIDN_ZIP_MODE off)
mark_as_advanced(OIDN_ZIP_MODE)

## MKL-DNN ####################################################################

# Configure MKL-DNN
set(MKLDNN_LIBRARY_TYPE "STATIC" CACHE INTERNAL "")
set(MKLDNN_THREADING "TBB" CACHE INTERNAL "")
set(MKLDNN_USE_MKL "NONE" CACHE INTERNAL "")
set(MKLDNN_VERBOSE OFF CACHE BOOL "")
set(WITH_EXAMPLE OFF CACHE INTERNAL "")
set(WITH_TEST OFF CACHE INTERNAL "")

# Add modified version of MKL-DNN
add_subdirectory(mkl-dnn EXCLUDE_FROM_ALL)

# Include some modules from MKL-DNN
include(mkl-dnn/cmake/Threading.cmake)
include(mkl-dnn/cmake/TBB.cmake)
include(mkl-dnn/cmake/platform.cmake)
include(mkl-dnn/cmake/SDL.cmake)

## Open Image Denoise Library #################################################

add_subdirectory(common EXCLUDE_FROM_ALL)
add_subdirectory(core)

## Open Image Denoise Examples ################################################

add_subdirectory(examples)

## Open Image Denoise Install and Packaging ###################################

include(install)
include(package)
