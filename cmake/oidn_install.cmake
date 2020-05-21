## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

## -----------------------------------------------------------------------------
## Install library
## -----------------------------------------------------------------------------

install(TARGETS ${PROJECT_NAME}
  EXPORT
    ${PROJECT_NAME}_Export
  ARCHIVE
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  # On Windows put the dlls into bin
  RUNTIME
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib
)

## -----------------------------------------------------------------------------
## Install headers
## -----------------------------------------------------------------------------

install(DIRECTORY include/OpenImageDenoise
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT devel
  PATTERN "*.in" EXCLUDE
)

## -----------------------------------------------------------------------------
## Install documentation
## -----------------------------------------------------------------------------

install(
  FILES
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/readme.pdf
    ${PROJECT_SOURCE_DIR}/CHANGELOG.md
    ${PROJECT_SOURCE_DIR}/LICENSE.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs-DNNL.txt
    ${PROJECT_SOURCE_DIR}/third-party-programs-TBB.txt
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  COMPONENT lib
)

## -----------------------------------------------------------------------------
## Install dependencies: TBB
## -----------------------------------------------------------------------------

if(OIDN_STATIC_LIB)
  install(TARGETS TBB EXPORT TBB_Export)
  install(EXPORT TBB_Export
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    FILE TBBConfig.cmake
    COMPONENT devel
  )
endif()

if(OIDN_ZIP_MODE)
  foreach(C IN ITEMS "tbb" "tbbmalloc")
    get_target_property(LIB_PATH TBB::${C} IMPORTED_LOCATION_RELEASE)
    if(WIN32)
      install(PROGRAMS ${LIB_PATH} DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT lib)
      get_target_property(IMPLIB_PATH TBB::${C} IMPORTED_IMPLIB_RELEASE)
      install(PROGRAMS ${IMPLIB_PATH} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib)
    else()
      install(PROGRAMS ${LIB_PATH} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib)
    endif()
  endforeach()
endif()

## -----------------------------------------------------------------------------
## Install CMake configuration files
## -----------------------------------------------------------------------------

install(EXPORT ${PROJECT_NAME}_Export
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  #NAMESPACE ${PROJECT_NAME}::
  FILE ${PROJECT_NAME}Config.cmake
  COMPONENT devel
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${PROJECT_NAME}ConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  COMPONENT devel
)
