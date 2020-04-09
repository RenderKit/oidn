#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

source scripts/unix_common.sh "$@"

cd $BUILD_DIR

# Create tar.gz file
make -j $THREADS package

cd $ROOT_DIR


