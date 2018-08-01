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

cmake_minimum_required(VERSION 2.8)
project(OpenImageDenoise)

set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
if (win32)
  if (NOT OIDN_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET)
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

# Modified version of MKL-DNN
add_subdirectory(mkl-dnn)

# Include some modules from MKL-DNN
include(mkl-dnn/cmake/platform.cmake)
include(mkl-dnn/cmake/TBB.cmake)
include(mkl-dnn/cmake/SDL.cmake)

# Generates C++ resource files from the specified binary blobs
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

# Set the include directories
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/mkl-dnn/src
  ${PROJECT_SOURCE_DIR}/mkl-dnn/src/common
  ${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu/xbyak
  ${TBB_INCLUDE_DIR}
)

# Add the OIDN common library
add_subdirectory(common)

# The OIDN library
set(SOURCES
  include/OpenImageDenoise/oidn.h
  include/OpenImageDenoise/oidn.hpp
  filters/common.h
  filters/device.h
  filters/device.cpp
  filters/buffer_view.h
  filters/filter.h
  filters/node.h
  filters/upsample.h
  filters/input_reorder.h
  filters/output_reorder.h
  filters/weights_reorder.h
  filters/network.h
  filters/network.cpp
  filters/autoencoder.h
  filters/autoencoder.cpp
  filters/api.cpp
)

generate_cpp_resources(WEIGHTS_SOURCES "oidn::Weights"
  weights/ae_ldr_albedo_normal.tza
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${WEIGHTS_SOURCES})
target_link_libraries(${PROJECT_NAME} common mkldnn ${TBB_LIBRARIES})

# OIDN example apps
add_subdirectory(examples)
