// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sycl_device.h"
#include "buffer.h"

namespace oidn {

  // SYCL buffer which can optionally own its data
  class SYCLBuffer : public Buffer
  {
  private:
    char* ptr;
    size_t byteSize;
    bool shared;
    Kind kind;
    Ref<SYCLDevice> device;

  public:
    SYCLBuffer(const Ref<SYCLDevice>& device, size_t size, Kind kind)
      : byteSize(size),
        shared(false),
        kind(kind),
        device(device)
    {
      ptr = allocData(size);
    }

    SYCLBuffer(const Ref<SYCLDevice>& device, void* data, size_t size)
      : ptr((char*)data),
        byteSize(size),
        shared(true),
        kind(Kind::Unknown),
        device(device)
    {
      if (data == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");
    }

    ~SYCLBuffer()
    {
      if (!shared)
        freeData(ptr);
    }

    char* data() override { return ptr; }
    const char* data() const override { return ptr; }
    size_t size() const override { return byteSize; }

    void* map(size_t offset, size_t size) override
    {
      if (offset + size > byteSize)
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      return ptr + offset;
    }

    void unmap(void* mappedPtr) override {}

    void resize(size_t newSize) override
    {
      if (shared)
        throw std::logic_error("shared buffers cannot be resized");

      freeData(ptr);
      ptr = allocData(newSize);
      byteSize = newSize;
    }

    Device* getDevice() override { return device.get(); }

  private:
    char* allocData(size_t size)
    {
      switch (kind)
      {
      case Kind::Host:
        return (char*)cl::sycl::aligned_alloc_host(memoryAlignment,
                                                   size,
                                                   device->getSYCLContext());

      case Kind::Device:
        return (char*)cl::sycl::aligned_alloc_device(memoryAlignment,
                                                     size,
                                                     device->getSYCLDevice(),
                                                     device->getSYCLContext());

      case Kind::Shared:
        return (char*)cl::sycl::aligned_alloc_shared(memoryAlignment,
                                                     size,
                                                     device->getSYCLDevice(),
                                                     device->getSYCLContext());

      default:
        throw Exception(Error::InvalidArgument, "invalid SYCL buffer type");
      }
    }
    
    void freeData(void* ptr)
    {
      cl::sycl::free(ptr, device->getSYCLContext());
    }
  };

} // namespace oidn
