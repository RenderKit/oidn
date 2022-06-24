## Copyright 2009-2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(APPLE)
  set(SDK_VERSION_COMMAND  xcrun -sdk  macosx --show-sdk-version)
  set(SDK_TARGET 11.0)

  execute_process(COMMAND  ${SDK_VERSION_COMMAND}
                  OUTPUT_VARIABLE SDK_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(SDK_TARGET VERSION_LESS SDK_VERSION)
    set(SDK_TARGET SDK_VERSION)
  endif()
  set(CMAKE_OSX_DEPLOYMENT_TARGET ${SDK_VERSION})
  set(OIDN_MPS ON)
endif()