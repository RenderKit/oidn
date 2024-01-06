// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/buffer.h"
#include "hip_engine.h"

OIDN_NAMESPACE_BEGIN

  class HIPExternalBuffer : public USMBuffer
  {
  public:
    HIPExternalBuffer(Engine* engine,
                      ExternalMemoryTypeFlag fdType,
                      int fd, size_t byteSize);

    HIPExternalBuffer(Engine* engine,
                      ExternalMemoryTypeFlag handleType,
                      void* handle, const void* name, size_t byteSize);

    ~HIPExternalBuffer();

  private:
    hipExternalMemory_t extMem;

    void init(const hipExternalMemoryHandleDesc& handleDesc);
  };

OIDN_NAMESPACE_END
