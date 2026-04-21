// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/semaphore.h"
#include "sycl_engine.h"

OIDN_NAMESPACE_BEGIN

  class SYCLExternalSemaphore : public Semaphore
  {
  public:
    SYCLExternalSemaphore(SYCLEngine* engine,
                          ExternalSemaphoreTypeFlags fdType,
                          int fd);

    SYCLExternalSemaphore(SYCLEngine* engine,
                          ExternalSemaphoreTypeFlags handleType,
                          void* handle, const void* name);

    ~SYCLExternalSemaphore();

    ExternalSemaphoreTypeFlags getType() const { return type; }
    sycl::ext::oneapi::experimental::external_semaphore getHandle() const { return extSem; }

  private:
    ExternalSemaphoreTypeFlags type;
    sycl::ext::oneapi::experimental::external_semaphore extSem;
    SYCLEngine* engine;
  };

OIDN_NAMESPACE_END
