#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

source scripts/unix_common.sh "$@"

cd $ROOT_DIR
mkdir -p $DEP_DIR
cd $DEP_DIR

# Set up TBB
OIDN_TBB_ROOT="${TBB_DIR}/mac/tbb"
RETRY_COUNTER=1
while [ 1 ]; do
  if [ -d "$OIDN_TBB_ROOT" ]; then
    break
  fi
  sleep 1
  if [ $RETRY_COUNTER -ge 10 ]; then
    echo "Cannot find TBB root at ${OIDN_TBB_ROOT}. Download TBB using scripts/download_tbb.sh."
    exit 1
  fi
  # Macos loses the NAS every now and then.
  echo "Could not find ${OIDN_TBB_ROOT}. Retrying..."
  ((RETRY_COUNTER++))
done

# Get the number of build threads
THREADS=`sysctl -n hw.physicalcpu`

# Create a clean build directory
cd $ROOT_DIR
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

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

