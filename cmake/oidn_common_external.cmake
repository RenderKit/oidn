## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Define cache variables for common paths which will be set by the main library build
set(OIDN_ROOT_BINARY_DIR "" CACHE PATH "Location of the main library build directory.")
set(OIDN_INSTALL_RPATH_PREFIX "" CACHE PATH "Prefix for the RPATH of installed binaries.")

if(NOT OIDN_ROOT_BINARY_DIR)
  message(FATAL_ERROR "OIDN_ROOT_BINARY_DIR is not set. The cache may have been deleted, please try building again.")
endif()

# Common
include(oidn_common)

# Import targets from the main library build directory
include(${OIDN_BUILD_TREE_EXPORT_FILE})