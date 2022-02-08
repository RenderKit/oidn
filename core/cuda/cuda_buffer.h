// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../buffer.h"
#include "cuda_device.h"

namespace oidn {

  struct CUDABufferAllocator
  {
    static void* allocate(const Ref<CUDADevice>& device, size_t size, MemoryKind kind);
    static void deallocate(const Ref<CUDADevice>& device, void* ptr, MemoryKind kind);
  };

  using CUDABuffer = USMBuffer<CUDADevice, CUDABufferAllocator>;

} // namespace oidn
