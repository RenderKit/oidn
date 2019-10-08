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

# Fail scripts if individual commands fail.
set -e
set -o pipefail

# Set up the compiler
if [ -z "${OIDN_C}" ] || [ -z "${OIDN_CXX}" ]; then
  COMPILER=icc
  if [ "$#" -ge 1 ]; then
    COMPILER=$1
  fi
  if [[ $COMPILER == icc ]]; then
    C_COMPILER=icc
    CXX_COMPILER=icpc
  elif [[ $COMPILER == clang ]]; then
    C_COMPILER=clang
    CXX_COMPILER=clang++
  elif [[ $COMPILER == gcc ]]; then
    C_COMPILER=gcc
    CXX_COMPILER=g++
  else
    echo "Error: unknown compiler"
    exit 1
  fi
else
  C_COMPILER="${OIDN_C}"
  CXX_COMPILER="${OIDN_CXX}"
fi

# Set up dependencies
if [ -z "${OIDN_ROOT_DIR}" ]; then
  ROOT_DIR=$PWD
else
  ROOT_DIR="${OIDN_ROOT_DIR}"
fi

BUILD_DIR=$ROOT_DIR/build_release
DEP_DIR=$ROOT_DIR/deps

source ${ROOT_DIR}/scripts/tbb_version.sh
if [ -z "${OIDN_TBB_DIR_UNIX}" ]; then
  TBB_DIR="$DEP_DIR/tbb/${TBB_VERSION}_${TBB_BUILD}"
else
  TBB_DIR="${OIDN_TBB_DIR_UNIX}/${TBB_VERSION}_${TBB_BUILD}"
fi

