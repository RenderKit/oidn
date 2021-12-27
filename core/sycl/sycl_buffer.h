// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../buffer.h"
#include "sycl_device.h"

namespace oidn {

  struct SYCLBufferAllocator
  {
    static void* allocate(const Ref<SYCLDevice>& device, size_t size, Buffer::Kind kind);
    static void deallocate(const Ref<SYCLDevice>& device, void* ptr, Buffer::Kind kind);
  };

  using SYCLBuffer = USMBuffer<SYCLDevice, SYCLBufferAllocator>;

} // namespace oidn
