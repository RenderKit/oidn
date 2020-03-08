## =============================================================================
## Copyright 2009-2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
## =============================================================================

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

