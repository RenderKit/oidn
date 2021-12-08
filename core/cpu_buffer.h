// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"

namespace oidn {

  struct CPUBufferAllocator
  {
    static void* allocate(const Ref<Device>& device, size_t size, Buffer::Kind kind)
    {
      return alignedMalloc(size);
    }

    static void deallocate(const Ref<Device>& device, void* ptr, Buffer::Kind kind)
    {
      alignedFree(ptr);
    }
  };

  using CPUBuffer = USMBuffer<Device, CPUBufferAllocator>;

} // namespace oidn
