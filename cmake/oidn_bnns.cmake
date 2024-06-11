## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if(NOT IOS)
  set(OIDN_APPLE_SDK_VERSION_MIN 11.0)
  set(OIDN_APPLE_SDK_VERSION_MAX 11.0)
else()
  set(OIDN_APPLE_SDK_VERSION_MIN 14.0)
  set(OIDN_APPLE_SDK_VERSION_MAX 14.0)
endif()

if(OIDN_APPLE_SDK_VERSION VERSION_LESS OIDN_APPLE_SDK_VERSION_MAX)
  message(FATAL_ERROR "Building with BNNS support requires Apple SDK version ${OIDN_APPLE_SDK_VERSION_MAX} or newer")
endif()