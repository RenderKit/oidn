## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

param(
[string]$COMPILER = "icc"
)

$ErrorActionPreference = 'Stop'

. scripts/windows_common.ps1 $COMPILER

Set-Location "${ROOT_DIR}/${BUILD_DIR}"

cmake -L `
      -D OIDN_ZIP_MODE=ON `
      ..

# Build.
cmake --build . `
      --config Release `
      --target PACKAGE

