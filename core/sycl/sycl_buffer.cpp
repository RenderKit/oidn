// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_buffer.h"

namespace oidn {

  void* SYCLBufferAllocator::allocate(const Ref<SYCLDevice>& device, size_t size, MemoryKind kind)
  {
    switch (kind)
    {
    case MemoryKind::Host:
      return sycl::aligned_alloc_host(memoryAlignment,
                                      size,
                                      device->getSYCLContext());

    case MemoryKind::Device:
      return sycl::aligned_alloc_device(memoryAlignment,
                                        size,
                                        device->getSYCLDevice(),
                                        device->getSYCLContext());

    case MemoryKind::Shared:
      return sycl::aligned_alloc_shared(memoryAlignment,
                                        size,
                                        device->getSYCLDevice(),
                                        device->getSYCLContext());

    default:
      throw Exception(Error::InvalidArgument, "invalid SYCL buffer type");
    }
  }

  void SYCLBufferAllocator::deallocate(const Ref<SYCLDevice>& device, void* ptr, MemoryKind kind)
  {
    sycl::free(ptr, device->getSYCLContext());
  }

} // namespace oidn
