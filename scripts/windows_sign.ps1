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

