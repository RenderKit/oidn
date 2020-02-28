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

macro(add_example EXAMPLE_NAME)
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp image_io.h image_io.cpp cli.h)
  target_link_libraries(${EXAMPLE_NAME} PRIVATE common ${PROJECT_NAME})
  install(TARGETS ${EXAMPLE_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT examples)
  find_library(OPENEXR_ILMIMF_LIBRARY NAMES IlmImf)
  if(OPENEXR_ILMIMF_LIBRARY)
    target_compile_definitions(${EXAMPLE_NAME} PRIVATE HAS_OPEN_EXR)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE ${OPENEXR_ILMIMF_LIBRARY})
  endif()
endmacro()

add_example(denoise)

