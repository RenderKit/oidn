#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

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

if [ -z "${OIDN_BUILD}" ]; then
  BUILD_TYPE=Release
  if [ "$#" -ge 2 ]; then
    BUILD_TYPE=$2
  fi
else
  BUILD_TYPE="${OIDN_BUILD}"
fi

# Set up dependencies
if [ -z "${OIDN_ROOT_DIR}" ]; then
  ROOT_DIR=$PWD
else
  ROOT_DIR="${OIDN_ROOT_DIR}"
fi

BUILD_DIR=$ROOT_DIR/build_`echo "$BUILD_TYPE" | awk '{print tolower($0)}'`
DEP_DIR=$ROOT_DIR/deps

source ${ROOT_DIR}/scripts/tbb_version.sh
if [ -z "${OIDN_TBB_DIR_UNIX}" ]; then
  TBB_DIR="$DEP_DIR/tbb/${TBB_VERSION}"
else
  TBB_DIR="${OIDN_TBB_DIR_UNIX}/${TBB_VERSION}"
fi

