## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

option(OIDN_ZIP_MODE off)
mark_as_advanced(OIDN_ZIP_MODE)

## -----------------------------------------------------------------------------
## Set install directories
## -----------------------------------------------------------------------------

include(GNUInstallDirs)

if(OIDN_ZIP_MODE)
  # Override install directories
  set(CMAKE_INSTALL_BINDIR bin)
  set(CMAKE_INSTALL_LIBDIR lib)
  set(CMAKE_INSTALL_DOCDIR doc)
endif()

## -----------------------------------------------------------------------------
## Set rpath
## -----------------------------------------------------------------------------

if(OIDN_ZIP_MODE)
  set(CMAKE_SKIP_INSTALL_RPATH OFF)
  if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
  else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
  endif()
else()
  # Normally we would use ${CMAKE_INSTALL_FULL_LIBDIR} but since the final install prefix is in
  # OIDN_INSTALL_RPATH_PREFIX for external projects, we need to determine the full path manually.
  if(CMAKE_INSTALL_LIBDIR)
    if(IS_ABSOLUTE ${CMAKE_INSTALL_LIBDIR})
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_LIBDIR}")
    else()
      set(CMAKE_INSTALL_RPATH "${OIDN_INSTALL_RPATH_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    endif()
  else()
    set(CMAKE_INSTALL_RPATH "${OIDN_INSTALL_RPATH_PREFIX}/lib")
  endif()
endif()

## -----------------------------------------------------------------------------
## Configure CPack
## -----------------------------------------------------------------------------

set(CPACK_PACKAGE_NAME OpenImageDenoise)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Intel(R) Open Image Denoise library")
set(CPACK_PACKAGE_VENDOR "Intel Corporation")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME oidn-${CPACK_PACKAGE_VERSION}${OIDN_VERSION_NOTE})
set(CPACK_VERBATIM_VARIABLES YES)

if(WIN32)
  # Windows specific settings
  if(OIDN_ARCH STREQUAL "X64")
    set(ARCH "x64")
    set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME} x64")
  elseif(OIDN_ARCH STREQUAL "ARM64")
    set(ARCH "arm64")
    set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME} ARM64")
  endif()

  set(CPACK_GENERATOR ZIP)
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.${ARCH}.windows")
  set(CPACK_MONOLITHIC_INSTALL 1)
else()
  if(OIDN_ARCH STREQUAL "X64")
    set(ARCH "x86_64")
  elseif(OIDN_ARCH STREQUAL "ARM64")
    set(ARCH "arm64")
  endif()

  if(APPLE)
    # macOS specific settings
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.${ARCH}.macos")
    set(CPACK_MONOLITHIC_INSTALL 1)
  else()
    # Linux specific settings
    #set(CPACK_GENERATOR RPM)
    #set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.${ARCH}.linux")
    #set(CPACK_MONOLITHIC_INSTALL 1)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.${ARCH}.linux")
    set(CPACK_MONOLITHIC_INSTALL 1)
  endif()
endif()
