## ======================================================================== ##
## Copyright 2009-2018 Intel Corporation                                    ##
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

project(OpenImageDenoise VERSION 0.5.0 LANGUAGES CXX)

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

## Helper Functions ###########################################################

# Generates C++ resource files from the specified binary blobs
find_package(PythonInterp REQUIRED)
function(generate_cpp_resources out_sources namespace)
  set(${out_sources})
  foreach(in_file ${ARGN})
    get_filename_component(in_file_we ${in_file} NAME_WE)
    get_filename_component(in_dir ${in_file} PATH)
    get_filename_component(in_path ${in_file} ABSOLUTE)
    set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/${in_dir})
    set(out_path ${out_dir}/${in_file_we}.cpp)
    list(APPEND ${out_sources} ${out_path})
    add_custom_command(
      OUTPUT ${out_path}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${out_dir}
      COMMAND ${PYTHON_EXECUTABLE}
      ARGS ${PROJECT_SOURCE_DIR}/scripts/blob2cpp.py ${in_path} -o ${out_path} -n ${namespace}
      DEPENDS ${in_path}
      COMMENT "Generating CXX resource object ${out_path}"
      VERBATIM)
  endforeach()
  set_source_files_properties(${${out_sources}} PROPERTIES GENERATED TRUE)
  set(${out_sources} ${${out_sources}} PARENT_SCOPE)
endfunction()

## Open Image Denoise Library #################################################

include_directories(
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/mkl-dnn/include
  ${PROJECT_SOURCE_DIR}/mkl-dnn/src
  ${PROJECT_SOURCE_DIR}/mkl-dnn/src/common
  ${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu/xbyak
  ${TBB_INCLUDE_DIR}
)

add_subdirectory(common EXCLUDE_FROM_ALL)

set(CORE_SOURCES
  include/OpenImageDenoise/oidn.h
  include/OpenImageDenoise/oidn.hpp
  core/api.cpp
  core/common.h
  core/math.h
  core/device.h
  core/device.cpp
  core/buffer.h
  core/image.h
  core/filter.h
  core/node.h
  core/input_reorder.h
  core/output_reorder.h
  core/weights_reorder.h
  core/transfer_function.h
  core/upsample.h
  core/network.h
  core/autoencoder.h
  core/autoexposure.h
)

set(CORE_SOURCES_SSE42
  core/network.cpp
  core/autoencoder.cpp
  core/autoexposure.cpp
)

generate_cpp_resources(WEIGHTS_SOURCES "oidn::weights"
  weights/rt_ldr.tza
  weights/rt_ldr_alb_nrm.tza
  weights/rt_hdr.tza
  weights/rt_hdr_alb_nrm.tza
)

add_library(${PROJECT_NAME} SHARED ${CORE_SOURCES} ${WEIGHTS_SOURCES})

add_library("${PROJECT_NAME}_sse42" STATIC ${CORE_SOURCES_SSE42})
set_target_properties("${PROJECT_NAME}_sse42" PROPERTIES COMPILE_FLAGS "${ISA_FLAGS_SSE42}")

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${TBB_LIBRARIES}
  PRIVATE
    "${PROJECT_NAME}_sse42"
    common mkldnn
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION "0")

## Open Image Denoise Examples ################################################

add_subdirectory(examples)

## Open Image Denoise Install and Packaging ###################################

include(install)
include(package)
