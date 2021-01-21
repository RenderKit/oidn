## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(NOT IOS)
  set(SDK_VERSION_COMMAND  xcrun -sdk  macosx --show-sdk-version)
  set(SDK_TARGET 11.0)
else()
  set(SDK_VERSION_COMMAND  xcrun -sdk  iphoneos --show-sdk-version)
  set(SDK_TARGET 14.0)
endif()

execute_process(COMMAND  ${SDK_VERSION_COMMAND}
                OUTPUT_VARIABLE SDK_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(SDK_VERSION VERSION_LESS SDK_TARGET)
  message(FATAL_ERROR "Apple SDK version " ${SDK_TARGET} " or above is required to use BNNS (current SDK version is " ${SDK_VERSION} ")")
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS SDK_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET ${SDK_TARGET})
endif()