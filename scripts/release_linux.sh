#!/bin/bash

## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set -e
scripts/linux_build.sh         "$@"
scripts/linux_check_symbols.sh "$@"
scripts/linux_package.sh       "$@"

