#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

source scripts/unix_common.sh "$@"

cd $BUILD_DIR

if [ -z $OIDN_SIGN_FILE_BINARY_APPLE ]; then
  echo "\$OIDN_SIGN_FILE_BINARY_APPLE is not set -- skipping sign stage."
else
  [ -x $OIDN_SIGN_FILE_BINARY_APPLE ] || exit
  $OIDN_SIGN_FILE_BINARY_APPLE libOpenImageDenoise.*.*.*.dylib
  $OIDN_SIGN_FILE_BINARY_APPLE denoise
  $OIDN_SIGN_FILE_BINARY_APPLE tests
fi

cd $ROOT_DIR

