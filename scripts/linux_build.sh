#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

source scripts/unix_common.sh "$@"

cd $ROOT_DIR
mkdir -p $DEP_DIR
cd $DEP_DIR

# Set up TBB
OIDN_TBB_ROOT="${TBB_DIR}/linux/tbb"
if [ ! -d "$OIDN_TBB_ROOT" ]; then
  echo "Cannot find TBB root at ${OIDN_TBB_ROOT}. Download TBB using scripts/download_tbb.sh."
  exit 1
fi

# Create a clean build directory
cd $ROOT_DIR
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

# Get the number of build threads
THREADS=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`


# Set compiler and release settings
cmake \
-D CMAKE_C_COMPILER:FILEPATH=$C_COMPILER \
-D CMAKE_CXX_COMPILER:FILEPATH=$CXX_COMPILER \
-D CMAKE_BUILD_TYPE=$BUILD_TYPE \
-D TBB_ROOT="${OIDN_TBB_ROOT}" .. \
-D OIDN_ZIP_MODE=ON \
..

# Build
make -j $THREADS preinstall VERBOSE=1

cd $ROOT_DIR

