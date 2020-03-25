## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(CTEST_PROJECT_NAME "Open Image Denoise")
set(TEST_REPOSITORY "https://github.com/OpenImageDenoise/oidn")

# Set up the build environment.
set(CTEST_SOURCE_DIRECTORY "${CTEST_SCRIPT_DIRECTORY}/..")
set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/build")
 
if (WIN32)
  string(REPLACE "\\" "/" CTEST_SOURCE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
  string(REPLACE "\\" "/" CTEST_BINARY_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()
 
message("CTEST_SOURCE_DIRECTORY = ${CTEST_SOURCE_DIRECTORY}")
message("CTEST_BINARY_DIRECTORY = ${CTEST_BINARY_DIRECTORY}")
message("http_proxy = $ENV{http_proxy}")
message("https_proxy = $ENV{https_proxy}")
message("no_proxy = $ENV{no_proxy}")
message("PATH = $ENV{PATH}")


# Set up the CDASH drop location for results.
site_name(HOSTNAME)
set(CTEST_SITE "${HOSTNAME}")
if(CTEST_DROP_SITE)
  set(CTEST_DROP_METHOD "http")
  set(CTEST_DROP_LOCATION "/CDash/submit.php?project=${CTEST_PROJECT_NAME}")
  set(CTEST_DROP_SITE_CDASH TRUE)
endif()

# Find system information.
find_program(UNAME NAMES uname)
macro(getuname name flag)
  exec_program("${UNAME}" ARGS "${flag}" OUTPUT_VARIABLE "${name}")
endmacro(getuname)

getuname(osname -s)
getuname(osrel  -r)
getuname(cpu    -m)

include(ProcessorCount)
ProcessorCount(numProcessors)
if(numProcessors EQUAL 0)
  SET(numProcessors 1)
endif()

# Reset.
ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})
ctest_start("Continuous")

# Set up the build command.
set(CTEST_BUILD_NAME "testbuild-${osname}-${cpu}")

# TODO: Set up common compile options here.
set(OIDN_CONFIGURE_OPTIONS "-G '$ENV{OIDN_BUILD_GENERATOR}'")
set(OIDN_BUILD_OPTIONS "")

# Build in release mode by default, but allow overrides.
if (DEFINED ENV{OIDN_BUILD_TYPE})
  set(OIDN_BUILD_TYPE "$ENV{OIDN_BUILD_TYPE}")
else()
  set(OIDN_BUILD_TYPE "Release")
endif()

if (WIN32)
  set(OIDN_CONFIGURE_OPTIONS "${OIDN_CONFIGURE_OPTIONS}  -T '$ENV{OIDN_TOOLSET}'")
  set(OIDN_BUILD_OPTIONS     "${OIDN_BUILD_OPTIONS} --config ${OIDN_BUILD_TYPE}")

else()
  if (DEFINED ENV{OIDN_CXX})
    set(OIDN_CONFIGURE_OPTIONS "${OIDN_CONFIGURE_OPTIONS} -D CMAKE_CXX_COMPILER=$ENV{OIDN_COMPILER_PATH}$ENV{OIDN_CXX}")
  endif ()
  if (DEFINED ENV{OIDN_C})
    set(OIDN_CONFIGURE_OPTIONS "${OIDN_CONFIGURE_OPTIONS} -D CMAKE_C_COMPILER=$ENV{OIDN_COMPILER_PATH}$ENV{OIDN_C}")
  endif ()

  set(OIDN_CONFIGURE_OPTIONS "${OIDN_CONFIGURE_OPTIONS} -D CMAKE_BUILD_TYPE=${OIDN_BUILD_TYPE}")
endif()

set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} ${OIDN_CONFIGURE_OPTIONS} ..")
set(CTEST_BUILD_COMMAND     "${CMAKE_COMMAND} --build . ${OIDN_BUILD_OPTIONS}")

# Configure the project using cmake
ctest_configure()

# Build the current configuration.
ctest_build(RETURN_VALUE RETVAL)
message("test.cmake: ctest_build returned ${RETVAL}")
if (NOT RETVAL EQUAL 0)
  message(FATAL_ERROR "test.cmake: build failed")
endif ()
