## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

param(
[string]$COMPILER = "icc"
)

$ErrorActionPreference = 'Stop'

. scripts/windows_common.ps1 $COMPILER

Set-Location "${ROOT_DIR}/${BUILD_DIR}"
Write-Host "Signing files in ${ROOT_DIR}/${BUILD_DIR} ..."

if (Test-Path Env:OIDN_SIGN_FILE_BINARY_WINDOWS) {
  &"$ENV:OIDN_SIGN_FILE_BINARY_WINDOWS" "-vv" "Release\*.dll"
  if (!$?) { Exit 1 }
  &"$ENV:OIDN_SIGN_FILE_BINARY_WINDOWS" "-vv" "Release\*.exe"
  if (!$?) { Exit 1 }
} else {
  Write-Host "OIDN_SIGN_FILE_BINARY_WINDOWS is not set -- skipping sign stage."
}

