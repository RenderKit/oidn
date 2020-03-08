#!/bin/bash

## =============================================================================
## Copyright 2009-2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
## =============================================================================

source scripts/unix_common.sh "$@"

BASE_URL="https://github.com/intel/tbb/releases/download/${TBB_VERSION}"

mkdir -p "${TBB_DIR}"

cd "${TBB_DIR}"
if [ ! -d "linux" ]; then
  mkdir -p linux
  cd linux
  LINUX_URL="${BASE_URL}/${TBB_BUILD}_lin.tgz"
  echo "Downloading ${LINUX_URL} ..."
  curl -L "${LINUX_URL}" | tar -xz
fi

cd "${TBB_DIR}"
if [ ! -d "mac" ]; then
  mkdir -p mac
  cd mac
  MACOS_URL="${BASE_URL}/${TBB_BUILD}_mac.tgz"
  echo "Downloading ${MACOS_URL} ..."
  curl -L "${MACOS_URL}" | tar -xz
fi

cd "${TBB_DIR}"
if [ ! -d "win" ]; then
  mkdir -p win
  cd win
  WINDOWS_URL="${BASE_URL}/${TBB_BUILD}_win.zip"
  echo "Downloading ${WINDOWS_URL} ..."
  curl -L "${WINDOWS_URL}" -o "${TBB_BUILD}_win.zip"
  unzip "${TBB_BUILD}_win.zip"
fi

