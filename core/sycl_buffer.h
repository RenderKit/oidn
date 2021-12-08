// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_device.h"
#include "buffer.h"

namespace oidn {

  struct SYCLBufferAllocator
  {
    static void* allocate(const Ref<SYCLDevice>& device, size_t size, Buffer::Kind kind)
    {
      switch (kind)
      {
      case Buffer::Kind::Host:
        return sycl::aligned_alloc_host(memoryAlignment,
                                        size,
                                        device->getSYCLContext());

      case Buffer::Kind::Device:
        return sycl::aligned_alloc_device(memoryAlignment,
                                          size,
                                          device->getSYCLDevice(),
                                          device->getSYCLContext());

      case Buffer::Kind::Shared:
        return sycl::aligned_alloc_shared(memoryAlignment,
                                          size,
                                          device->getSYCLDevice(),
                                          device->getSYCLContext());

      default:
        throw Exception(Error::InvalidArgument, "invalid SYCL buffer type");
      }
    }

    static void deallocate(const Ref<SYCLDevice>& device, void* ptr, Buffer::Kind kind)
    {
      sycl::free(ptr, device->getSYCLContext());
    }
  };

  using SYCLBuffer = USMBuffer<SYCLDevice, SYCLBufferAllocator>;

} // namespace oidn
