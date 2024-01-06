// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/buffer.h"
#include "cuda_engine.h"

OIDN_NAMESPACE_BEGIN

  class CUDAExternalBuffer : public USMBuffer
  {
  public:
    CUDAExternalBuffer(Engine* engine,
                       ExternalMemoryTypeFlag fdType,
                       int fd, size_t byteSize);

    CUDAExternalBuffer(Engine* engine,
                       ExternalMemoryTypeFlag handleType,
                       void* handle, const void* name, size_t byteSize);

    ~CUDAExternalBuffer();

  private:
    cudaExternalMemory_t extMem;

    void init(const cudaExternalMemoryHandleDesc& handleDesc);
  };

OIDN_NAMESPACE_END
