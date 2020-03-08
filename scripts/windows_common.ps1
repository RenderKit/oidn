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

Write-Host "Running on $env:computername"
if ((Test-Path Env:OIDN_GENERATOR) -and (Test-Path Env:OIDN_TOOLCHAIN)) {
  $GENERATOR = $Env:OIDN_GENERATOR
  $TOOLCHAIN = $Env:OIDN_TOOLCHAIN
}
else {
  if ($COMPILER -eq "icc") {
    $GENERATOR = "Visual Studio 15 2017 Win64"
    $TOOLCHAIN = "Intel C++ Compiler 19.0"
  } elseif ($COMPILER -eq "msvc") {
    $GENERATOR = "Visual Studio 15 2017 Win64"
    $TOOLCHAIN = ""
  } else {
    Write-Host "Invalid compiler: $COMPILER"
    exit 1
  }
}

if (Test-Path Env:OIDN_ROOT_DIR) {
  $ROOT_DIR = "$Env:OIDN_ROOT_DIR"
} else {
  $ROOT_DIR = Convert-Path "."
}
$DEP_DIR="$ROOT_DIR\deps"

# Read TBB version info from the shared script, and make sure TBB is there.
Get-Content "$ROOT_DIR\scripts\tbb_version.sh" | Where-Object {
  -not ([String]::IsNullOrEmpty($_.Trim()) -or $_-match"^#.*")
} | Foreach-Object {
  $var = $_.Split('=')
  New-Variable -Name $var[0] -Value $var[1]
}
Write-Host "TBB version is $TBB_VERSION"
Write-Host "TBB build is $TBB_BUILD"

if (Test-Path Env:OIDN_TBB_DIR_WINDOWS) {
  $TBB_DIR = "$Env:OIDN_TBB_DIR_WINDOWS\${TBB_VERSION}_${TBB_BUILD}\win\${TBB_BUILD}"
} else {
  $TBB_DIR = "${DEP_DIR}\tbb\${TBB_VERSION}_${TBB_BUILD}\win\${TBB_BUILD}"
}
Write-Host "Expecting to find TBB at $TBB_DIR"

# Clean up the build directory.
$BUILD_DIR = "build_release"

