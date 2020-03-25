## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

param(
[string]$COMPILER = "icc"
)

$ErrorActionPreference = 'Stop'

scripts/windows_build.ps1 $COMPILER
scripts/windows_sign.ps1 $COMPILER
scripts/windows_package.ps1 $COMPILER

