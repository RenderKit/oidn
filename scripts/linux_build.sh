#!/bin/bash

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

source scripts/unix_common.sh "$@"

cd $ROOT_DIR
mkdir -p $DEP_DIR
cd $DEP_DIR

# Set up TBB
OIDN_TBB_ROOT="${TBB_DIR}/linux/${TBB_BUILD}"
if [ ! -d "$OIDN_TBB_ROOT" ]; then
  echo "Cannot find tbb root at ${OIDN_TBB_ROOT}. Download tbb using scripts/download_tbb.sh."
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
-D TBB_ROOT="${OIDN_TBB_ROOT}" .. \
..

# Build
make -j $THREADS preinstall VERBOSE=1

cd $ROOT_DIR

