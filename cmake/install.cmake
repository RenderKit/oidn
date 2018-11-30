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

set(CMAKE_INSTALL_LIBDIR lib)

install(TARGETS ${PROJECT_NAME}
  EXPORT
    ${PROJECT_NAME}_Export COMPONENT devel
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  # On Windows put the dlls into bin
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
  INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT devel
)

# Use full absolute path as install name
if(NOT OIDN_ZIP_MODE)
  set(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_FULL_LIBDIR})
else()
  if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
  else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
  endif()
endif()

# Install headers
install(DIRECTORY include/OpenImageDenoise DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT devel)

# Install documentation
set(CMAKE_INSTALL_DOCDIR doc)
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE.txt DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT lib)
install(FILES ${PROJECT_SOURCE_DIR}/CHANGELOG.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT lib)
install(FILES ${PROJECT_SOURCE_DIR}/README.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT lib)

# Install TBB
if(OIDN_ZIP_MODE)
  if(WIN32)
    install(PROGRAMS ${TBB_BINDIR}/tbb.dll ${TBB_BINDIR}/tbbmalloc.dll DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib)
    install(PROGRAMS ${TBB_LIBDIR}/tbb.lib ${TBB_LIBDIR}/tbbmalloc.lib DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib)
  elseif(APPLE)
    install(PROGRAMS ${TBB_ROOT}/lib/libtbb.dylib ${TBB_ROOT}/lib/libtbbmalloc.dylib DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib)
  else()
    install(PROGRAMS ${TBB_ROOT}/lib/intel64/gcc4.4/libtbb.so.2 ${TBB_ROOT}/lib/intel64/gcc4.4/libtbbmalloc.so.2 DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib)
  endif()
endif()

install(EXPORT ${PROJECT_NAME}_Export
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  #NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Config.cmake
  COMPONENT devel
)
