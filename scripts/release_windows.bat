@echo off

rem ======================================================================== rem
rem Copyright 2009-2019 Intel Corporation                                    rem
rem                                                                          rem
rem Licensed under the Apache License, Version 2.0 (the "License");          rem
rem you may not use this file except in compliance with the License.         rem
rem You may obtain a copy of the License at                                  rem
rem                                                                          rem
rem     http://www.apache.org/licenses/LICENSE-2.0                           rem
rem                                                                          rem
rem Unless required by applicable law or agreed to in writing, software      rem
rem distributed under the License is distributed on an "AS IS" BASIS,        rem
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. rem
rem See the License for the specific language governing permissions and      rem
rem limitations under the License.                                           rem
rem ======================================================================== rem

setlocal

rem Set up the compiler
set COMPILER=icc
if not "%1" == "" (
  set COMPILER=%1
)
if %COMPILER% == icc (
  set TOOLSET="Intel C++ Compiler 18.0"
) else if %COMPILER% == msvc (
  set TOOLSET=""
) else (
  echo Error: unknown compiler
  exit /b 1
)

rem Set up dependencies
set ROOT_DIR=%cd%
set DEP_DIR=%ROOT_DIR%\deps

rem Check if TBB is there.
for /f "delims== tokens=1,2" %%G in (%ROOT_DIR%\scripts\tbb_version.sh) do set %%G=%%H

if "%OIDN_TBB_DIR_WINDOWS%"=="" (
  set TBB_DIR="%DEP_DIR%\tbb\%TBB_VERSION%_%TBB_BUILD%\win\%TBB_BUILD%"
) else (
  set TBB_DIR="%OIDN_TBB_DIR_WINDOWS%\%TBB_VERSION%_%TBB_BUILD%\win\%TBB_BUILD%"
)

if not exist %TBB_DIR% (
  echo Error: %TBB_DIR% is missing
  exit /b 1
)

rem Create a clean build directory
cd %ROOT_DIR%
rmdir /s /q build_release 2> NUL
mkdir build_release
cd build_release

rem Set compiler and release settings
cmake -L ^
-G "Visual Studio 15 2017 Win64" ^
-T %TOOLSET% ^
-D TBB_ROOT=%TBB_DIR% ^
..
if %ERRORLEVEL% geq 1 exit /b %ERRORLEVEL%

rem Create zip file
cmake -D OIDN_ZIP_MODE=ON ..
cmake --build . --config Release --target PACKAGE -- /m /nologo
if %ERRORLEVEL% geq 1 exit /b %ERRORLEVEL%

cd ..
