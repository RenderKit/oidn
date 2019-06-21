param(
[string]$COMPILER = "icc"
)

$ErrorActionPreference = 'Stop'

. scripts/windows_common.ps1 $COMPILER

Set-Location "${ROOT_DIR}/${BUILD_DIR}"

# Build.
cmake --build . `
      --config Release `
      --target PACKAGE

