// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "device.h"

namespace oidn {

  class Device;

  // Generic buffer object
  class Buffer : public RefCount
  {
  public:
    virtual char* data() = 0;
    virtual const char* data() const = 0;
    virtual size_t size() const = 0;

    virtual void* map(size_t offset, size_t size) = 0;
    virtual void unmap(void* mappedPtr) = 0;

    // Resizes the buffer discarding its current contents
    virtual void resize(size_t newSize)
    {
      throw std::logic_error("resizing is not supported");
    }

    virtual Device* getDevice() = 0;
  };

  // Buffer which may or may not own its data
  class CPUBuffer : public Buffer
  {
  private:
    char* ptr;
    size_t byteSize;
    bool shared;
    Ref<Device> device;

  public:
    CPUBuffer(const Ref<Device>& device, size_t size)
      : ptr(alloc(size)),
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
        alignedFree(ptr);
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

      alignedFree(ptr);
      ptr = alloc(newSize);
      byteSize = newSize;
    }

    Device* getDevice() override { return device.get(); }

  private:
    static char* alloc(size_t size)
    {
      return (char*)alignedMalloc(size);
    }
  };

  // Memory object backed by a buffer
  struct Memory
  {
    Ref<Buffer> buffer;  // buffer containing the data
    size_t bufferOffset; // offset in the buffer

    Memory() : bufferOffset(0) {}
    virtual ~Memory() = default;

    Memory(const Ref<Buffer>& buffer, size_t bufferOffset = 0)
      : buffer(buffer),
        bufferOffset(bufferOffset) {}

    // If the buffer gets reallocated, this must be called to update the internal pointer
    virtual void updatePtr() = 0;
  };

} // namespace oidn
