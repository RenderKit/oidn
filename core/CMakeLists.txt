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

set(CORE_SOURCES
  api.cpp
  common.h
  math.h
  device.h
  device.cpp
  buffer.h
  image.h
  filter.h
  node.h
  input_reorder.h
  output_reorder.h
  weights_reorder.h
  tone_mapping.h
  upsample.h
  network.h
  autoencoder.h
)

set(CORE_SOURCES_SSE42
  network.cpp
  autoencoder.cpp
  tone_mapping.cpp
)

generate_cpp_resources(WEIGHTS_SOURCES "oidn::weights"
  ../weights/rt_ldr.tza
  ../weights/rt_ldr_alb.tza
  ../weights/rt_ldr_alb_nrm.tza
  ../weights/rt_hdr.tza
  ../weights/rt_hdr_alb.tza
  ../weights/rt_hdr_alb_nrm.tza
)

add_library(${PROJECT_NAME} SHARED ${CORE_SOURCES} ${WEIGHTS_SOURCES})

add_library("${PROJECT_NAME}_sse42" STATIC ${CORE_SOURCES_SSE42})
target_link_libraries("${PROJECT_NAME}_sse42" PUBLIC common)
set_target_properties("${PROJECT_NAME}_sse42" PROPERTIES COMPILE_FLAGS "${ISA_FLAGS_SSE42}")
target_include_directories("${PROJECT_NAME}_sse42"
  PUBLIC
    ${PROJECT_SOURCE_DIR}/mkl-dnn/include
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src/common
    ${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu/xbyak
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${TBB_LIBRARIES}
  PRIVATE
    "${PROJECT_NAME}_sse42"
    common mkldnn
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION "0")
