## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

param(
[string]$COMPILER = "icc"
)

$ErrorActionPreference = 'Stop'

. scripts/windows_common.ps1 $COMPILER

# Clean up the build directory.
Set-Location "$ROOT_DIR"
$BUILD_DIR = "build_release"
if (Test-Path "$BUILD_DIR") {
  Get-ChildItem "$BUILD_DIR" -Recurse | Remove-Item
  Remove-Item "$BUILD_DIR"
}
New-Item -Path $ROOT_DIR -Name "$BUILD_DIR" -ItemType directory
Set-Location "${ROOT_DIR}/${BUILD_DIR}"

cmake --version


if (Test-Path Env:OIDN_ISPC_EXECUTABLE_WINDOWS) {
  cmake -L `
        -G "${GENERATOR}" `
        -T "${TOOLCHAIN}" `
        -D TBB_ROOT="${TBB_DIR}" `
        -D ISPC_EXECUTABLE="${Env:OIDN_ISPC_EXECUTABLE_WINDOWS}" `
        ..
} else {
  cmake -L `
        -G "${GENERATOR}" `
        -T "${TOOLCHAIN}" `
        -D TBB_ROOT=${TBB_DIR} `
        ..
}

if (!$?) { Exit $LASTEXITCODE }

# Build.
cmake --build . `
      --config Release `
      --target ALL_BUILD

if (!$?) { Exit $LASTEXITCODE }

