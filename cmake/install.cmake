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

# TODO: any specializations for macOS/Windows?

install(TARGETS ${PROJECT_NAME}
  EXPORT
    ${PROJECT_NAME}_Export
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/OpenImageDenoise DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ${PROJECT_NAME}_Export
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  #NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Config.cmake
)
