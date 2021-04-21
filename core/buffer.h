// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "device.h"

namespace oidn {

  class Device;

  // Buffer which may or may not own its data
  class Buffer : public RefCount
  {
  private:
    char* ptr;
    size_t byteSize;
    bool shared;
    Ref<Device> device;

  public:
    __forceinline Buffer(const Ref<Device>& device, size_t size)
      : ptr((char*)alignedMalloc(size, 128)),
        byteSize(size),
        shared(false),
        device(device) {}

    __forceinline Buffer(const Ref<Device>& device, void* data, size_t size)
      : ptr((char*)data),
        byteSize(size),
        shared(true),
        device(device)
    {
      if (data == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");
    }

    __forceinline ~Buffer()
    {
      if (!shared)
        alignedFree(ptr);
    }

    __forceinline char* data() { return ptr; }
    __forceinline const char* data() const { return ptr; }
    __forceinline size_t size() const { return byteSize; }

    void* map(size_t offset, size_t size)
    {
      if (offset + size > byteSize)
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      return ptr + offset;
    }

    void unmap(void* mappedPtr) {}

    Device* getDevice() { return device.get(); }
  };

} // namespace oidn
