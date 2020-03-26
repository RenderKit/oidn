## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(NOT OPENIMAGEIO_ROOT)
  set(OPENIMAGEIO_ROOT $ENV{OPENIMAGEIO_ROOT})
endif()
if(NOT OPENIMAGEIO_ROOT)
  set(OPENIMAGEIO_ROOT $ENV{OPENIMAGEIOROOT})
endif()

# detect changed OPENIMAGEIO_ROOT
if(NOT OPENIMAGEIO_ROOT STREQUAL OPENIMAGEIO_ROOT_LAST)
  unset(OPENIMAGEIO_INCLUDE_DIR CACHE)
  unset(OPENIMAGEIO_LIBRARY CACHE)
endif()

find_path(OPENIMAGEIO_ROOT include/OpenImageIO/imageio.h
  DOC "Root of OpenImageIO installation"
  HINTS ${OPENIMAGEIO_ROOT}
  PATHS
    ${PROJECT_SOURCE_DIR}/oiio
    /usr/local
    /usr
    /
)

find_path(OPENIMAGEIO_INCLUDE_DIR OpenImageIO/imageio.h PATHS ${OPENIMAGEIO_ROOT}/include NO_DEFAULT_PATH)
set(OPENIMAGEIO_HINTS
  HINTS
    ${OPENIMAGEIO_ROOT}
  PATH_SUFFIXES
    /lib
    /lib64
    /lib-vc2015
)
set(OPENIMAGEIO_PATHS PATHS /usr/lib /usr/lib64 /lib /lib64)
find_library(OPENIMAGEIO_LIBRARY OpenImageIO ${OPENIMAGEIO_HINTS} ${OPENIMAGEIO_PATHS})

set(OPENIMAGEIO_ROOT_LAST ${OPENIMAGEIO_ROOT} CACHE INTERNAL "Last value of OPENIMAGEIO_ROOT to detect changes")

set(OPENIMAGEIO_ERROR_MESSAGE "OpenImageIO not found in your environment. You can 1) install
                              via your OS package manager, or 2) install it
                              somewhere on your machine and point OPENIMAGEIO_ROOT to it.")

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenImageIO
  ${OPENIMAGEIO_ERROR_MESSAGE}
  OPENIMAGEIO_INCLUDE_DIR OPENIMAGEIO_LIBRARY
)

if(OPENIMAGEIO_FOUND)
  set(OPENIMAGEIO_INCLUDE_DIRS ${OPENIMAGEIO_INCLUDE_DIR})
  set(OPENIMAGEIO_LIBRARIES ${OPENIMAGEIO_LIBRARY})

  add_library(OpenImageIO::OpenImageIO UNKNOWN IMPORTED)
  set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OPENIMAGEIO_INCLUDE_DIRS}")
  set_property(TARGET OpenImageIO::OpenImageIO APPEND PROPERTY
    IMPORTED_LOCATION "${OPENIMAGEIO_LIBRARIES}")
endif()

mark_as_advanced(OPENIMAGEIO_INCLUDE_DIR)
mark_as_advanced(OPENIMAGEIO_LIBRARY)
