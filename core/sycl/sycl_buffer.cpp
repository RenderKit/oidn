// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_buffer.h"

namespace oidn {

  void* SYCLBufferAllocator::allocate(const Ref<SYCLDevice>& device, size_t size, Buffer::Kind kind)
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

  void SYCLBufferAllocator::deallocate(const Ref<SYCLDevice>& device, void* ptr, Buffer::Kind kind)
  {
    sycl::free(ptr, device->getSYCLContext());
  }

} // namespace oidn
