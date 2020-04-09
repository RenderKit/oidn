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
  Get-ChildItem "${ROOT_DIR}\${BUILD_DIR}" -Filter oidn-*.zip |
  Foreach-Object {
    Write-Host "Working on $zip"
    $zip = $_.FullName
    expand-archive -force -path "$zip" -destinationpath "unzipped"
    Write-Host "Looping..."
    Get-ChildItem "unzipped" |
    Foreach-Object {
      $dir = $_.FullName
      Write-Host "looking at $dir"
      Get-ChildItem "$dir\bin" -Filter *.dll |
      Foreach-Object {
        $file = $_.FullName
        &"$ENV:OIDN_SIGN_FILE_BINARY_WINDOWS" "-q" "-vv" "$file"
        if (!$?) { Exit 1 }
      }
      Get-ChildItem "$dir\bin" -Filter *.exe |
      Foreach-Object {
        $file = $_.FullName
        &"$ENV:OIDN_SIGN_FILE_BINARY_WINDOWS" "-q" "-vv" "$file"
        if (!$?) { Exit 1 }
      }
    }
    compress-archive -force -path "unzipped/oidn-*" -destinationpath "$zip"
    Remove-Item -Recurse -Force "${ROOT_DIR}/${BUILD_DIR}/unzipped"
  }
} else {
  Write-Host "OIDN_SIGN_FILE_BINARY_WINDOWS is not set -- skipping sign stage."
}

Set-Location "${ROOT_DIR}"

