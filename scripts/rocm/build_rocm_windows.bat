@rem Copyright 2023 Intel Corporation
@rem SPDX-License-Identifier: Apache-2.0

@echo off

set HIP_BRANCH=rocm-5.5.0

if [%1]==[] (
  echo Download, build and install ROCm on Windows ^(only HIP compiler without hiprtc^)
  echo Usage: %~nx0 ^<install_dir^>
  exit /b 1
)

if exist %1 (
  echo Error: Install directory already exists
  exit /b 1
)

set INSTALL_DIR=%1
set WORK_DIR=%cd%

rem Check whether the working directory is clean
if exist %WORK_DIR%\hip              goto work_dir_dirty
if exist %WORK_DIR%\hipamd           goto work_dir_dirty
if exist %WORK_DIR%\llvm-project     goto work_dir_dirty
if exist %WORK_DIR%\ROCm-Device-Libs goto work_dir_dirty

goto work_dir_clean
:work_dir_dirty
echo Error: Working directory is not clean
exit /b 1
:work_dir_clean

rem Check whether the required tools are in PATH
if "%VSINSTALLDIR%"=="" (
  echo Error: Visual Studio environment not set, please run from a Visual Studio command prompt
  exit /b 1
)

where cmake >nul 2>nul
if %errorlevel% neq 0 (
  echo Error: CMake not found, please install and add to PATH
  exit /b 1
)

where ninja >nul 2>nul
if %errorlevel% neq 0 (
  echo Error: Ninja not found, please install and add to PATH
  exit /b 1
)

where perl >nul 2>nul
if %errorlevel% neq 0 (
  echo Error: Perl not found, please install ^(e.g. Strawberry Perl^) and add to PATH
  exit /b 1
)

set SCRIPT_DIR=%~dp0
if %SCRIPT_DIR:~-1%==\ set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

rem Replace backslashes with forward slashes in paths we pass to CMake
set INSTALL_DIR=%INSTALL_DIR:\=/%
set WORK_DIR=%WORK_DIR:\=/%

rem Get the source code from GitHub
git clone --depth 1 -b %HIP_BRANCH% https://github.com/ROCm-Developer-Tools/hip.git
git clone --depth 1 -b %HIP_BRANCH% https://github.com/ROCm-Developer-Tools/hipamd.git
git clone --depth 1 -b %HIP_BRANCH% https://github.com/RadeonOpenCompute/llvm-project.git
git clone --depth 1 -b %HIP_BRANCH% https://github.com/RadeonOpenCompute/ROCm-Device-Libs.git

rem Build and install hipamd
cd hipamd
git apply --ignore-whitespace %SCRIPT_DIR%\rocm-hipamd.patch
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DHIP_COMMON_DIR="%WORK_DIR%/hip" -DHIP_PLATFORM=amd -DUSE_PROF_API=OFF -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ..
cmake --build . --target install
cd ..\..

rem Build and install llvm-project
cd llvm-project
git apply --ignore-whitespace %SCRIPT_DIR%\rocm-llvm-project.patch
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;compiler-rt" -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ..\llvm
cmake --build . --target install
cd ..\..

rem Build and install ROCm-Device-Libs
mkdir ROCm-Device-Libs\build
cd ROCm-Device-Libs\build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%WORK_DIR%/llvm-project/build" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ..
cmake --build . --target install
cd ..\..

echo ROCm installation complete