// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"

namespace oidn {

  // CPU buffer which can optionally own its data
  class CPUBuffer : public Buffer
  {
  private:
    char* ptr;
    size_t byteSize;
    bool shared;
    Ref<Device> device;

  public:
    CPUBuffer(const Ref<Device>& device, size_t size)
      : ptr(allocData(size)),
        byteSize(size),
        shared(false),
        device(device) {}

    CPUBuffer(const Ref<Device>& device, void* data, size_t size)
      : ptr((char*)data),
        byteSize(size),
        shared(true),
        device(device)
    {
      if (data == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");
    }

    ~CPUBuffer()
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
      return (char*)alignedMalloc(size);
    }

    void freeData(void* ptr)
    {
      alignedFree(ptr);
    }
  };

} // namespace oidn
