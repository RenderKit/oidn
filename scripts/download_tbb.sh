#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

source scripts/unix_common.sh "$@"

BASE_URL="https://github.com/intel/tbb/releases/download/v${TBB_VERSION}/tbb-${TBB_VERSION}"

mkdir -p "${TBB_DIR}"

cd "${TBB_DIR}"
if [ ! -d "linux" ]; then
  mkdir -p linux
  cd linux
  LINUX_URL="${BASE_URL}-lin.tgz"
  echo "Downloading ${LINUX_URL} ..."
  curl -L "${LINUX_URL}" | tar -xz
fi

cd "${TBB_DIR}"
if [ ! -d "mac" ]; then
  mkdir -p mac
  cd mac
  MACOS_URL="${BASE_URL}-mac.tgz"
  echo "Downloading ${MACOS_URL} ..."
  curl -L "${MACOS_URL}" | tar -xz
fi

cd "${TBB_DIR}"
if [ ! -d "win" ]; then
  mkdir -p win
  cd win
  WINDOWS_URL="${BASE_URL}-win.zip"
  echo "Downloading ${WINDOWS_URL} ..."
  curl -L "${WINDOWS_URL}" -o "tbb-${TBB_VERSION}-win.zip"
  unzip "tbb-${TBB_VERSION}-win.zip"
  rm "tbb-${TBB_VERSION}-win.zip"
fi

