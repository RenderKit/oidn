param(
[string]$COMPILER = "icc"
)

$ErrorActionPreference = 'Stop'

scripts/windows_build.ps1 $COMPILER
scripts/windows_sign.ps1 $COMPILER
scripts/windows_package.ps1 $COMPILER

