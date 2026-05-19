// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/semaphore.h"
#include "hip_engine.h"

OIDN_NAMESPACE_BEGIN

  class HIPExternalSemaphore : public Semaphore
  {
  public:
    HIPExternalSemaphore(Engine* engine,
                         ExternalSemaphoreTypeFlags fdType,
                         int fd);

    HIPExternalSemaphore(Engine* engine,
                         ExternalSemaphoreTypeFlags handleType,
                         void* handle, const void* name);

    ~HIPExternalSemaphore();

    ExternalSemaphoreTypeFlags getType() const { return type; }
    hipExternalSemaphore_t getHandle() const { return extSem; }

  private:
    ExternalSemaphoreTypeFlags type;
    hipExternalSemaphore_t extSem;

    void init(const hipExternalSemaphoreHandleDesc& handleDesc);
  };

OIDN_NAMESPACE_END