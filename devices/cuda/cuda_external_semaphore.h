// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/semaphore.h"
#include "cuda_engine.h"

OIDN_NAMESPACE_BEGIN

  class CUDAExternalSemaphore : public Semaphore
  {
  public:
    CUDAExternalSemaphore(Engine* engine,
                          ExternalSemaphoreTypeFlags fdType,
                          int fd);

    CUDAExternalSemaphore(Engine* engine,
                          ExternalSemaphoreTypeFlags handleType,
                          void* handle, const void* name);

    ~CUDAExternalSemaphore();

    ExternalSemaphoreTypeFlags getType() const { return type; }
    cudaExternalSemaphore_t getHandle() const { return extSem; }

  private:
    ExternalSemaphoreTypeFlags type;
    cudaExternalSemaphore_t extSem;

    void init(const cudaExternalSemaphoreHandleDesc& handleDesc);
  };

OIDN_NAMESPACE_END