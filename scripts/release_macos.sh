#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set -e
scripts/macos_build.sh    "$@"
scripts/macos_sign.sh     "$@"
scripts/macos_package.sh  "$@"

