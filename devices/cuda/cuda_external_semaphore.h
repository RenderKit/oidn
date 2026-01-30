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
                          ExternalSemaphoreTypeFlag fdType,
                          int fd);

    CUDAExternalSemaphore(Engine* engine,
                          ExternalSemaphoreTypeFlag handleType,
                          void* handle, const void* name);

    ~CUDAExternalSemaphore();

    ExternalSemaphoreTypeFlag getType() const { return type; }
    cudaExternalSemaphore_t getHandle() const { return extSem; }

  private:
    ExternalSemaphoreTypeFlag type;
    cudaExternalSemaphore_t extSem;

    void init(const cudaExternalSemaphoreHandleDesc& handleDesc);
  };

OIDN_NAMESPACE_END