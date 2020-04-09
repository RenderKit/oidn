#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

source scripts/unix_common.sh "$@"

cd $BUILD_DIR

if [ -z $OIDN_SIGN_FILE_BINARY_APPLE ]; then
  echo "\$OIDN_SIGN_FILE_BINARY_APPLE is not set -- skipping sign stage."
else
  [ -x $OIDN_SIGN_FILE_BINARY_APPLE ] || exit

  for PACKAGE in *.tar.gz; do
    tar -xf $PACKAGE
    rm $PACKAGE
    PACKAGE_DIR="${PACKAGE%.tar.gz}"
    for FILE in $PACKAGE_DIR/bin/* $PACKAGE_DIR/lib/*.*.*.*.dylib; do
      $OIDN_SIGN_FILE_BINARY_APPLE -q $FILE
    done
    tar -czf $PACKAGE $PACKAGE_DIR
    rm -rf $PACKAGE_DIR
  done
fi

cd $ROOT_DIR

