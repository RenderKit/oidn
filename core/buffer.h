// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  class Device;

  // Memory allocation kind
  enum class MemoryKind
  {
    Host,
    Device,
    Shared,
    Unknown
  };

  // Generic buffer object
  class Buffer : public RefCount
  {
  public:
    virtual char* getData() = 0;
    virtual const char* getData() const = 0;
    virtual size_t getByteSize() const = 0;

    virtual void* map(size_t offset, size_t size) = 0;
    virtual void unmap(void* mappedPtr) = 0;

    // Resizes the buffer discarding its current contents
    virtual void resize(size_t newSize)
    {
      throw std::logic_error("resizing is not supported");
    }

    virtual Device* getDevice() = 0;
  };

  // Unified shared memory based buffer object
  template<typename DeviceT, typename BufferAllocatorT>
  class USMBuffer : public Buffer
  {
  public:
    USMBuffer(const Ref<DeviceT>& device, size_t byteSize, MemoryKind kind)
      : ptr(nullptr),
        byteSize(byteSize),
        shared(false),
        kind(kind),
        device(device)
    {
      ptr = (char*)allocator.allocate(device, byteSize, kind);
    }

    USMBuffer(const Ref<DeviceT>& device, void* data, size_t byteSize)
      : ptr((char*)data),
        byteSize(byteSize),
        shared(true),
        kind(MemoryKind::Unknown),
        device(device)
    {
      if (ptr == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");
    }

    ~USMBuffer()
    {
      if (!shared)
        allocator.deallocate(device, ptr, kind);
    }

    char* getData() override { return ptr; }
    const char* getData() const override { return ptr; }
    size_t getByteSize() const override { return byteSize; }

    void* map(size_t offset, size_t size) override
    {
      if (offset + size > byteSize)
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      return ptr + offset;
    }

    void unmap(void* mappedPtr) override {}

    // Resizes the buffer discarding its current contents
    void resize(size_t newSize) override
    {
      if (shared)
        throw std::logic_error("shared buffers cannot be resized");

      allocator.deallocate(device, ptr, kind);
      ptr = (char*)allocator.allocate(device, newSize, kind);
      byteSize = newSize;
    }

    Device* getDevice() override { return device.get(); }

  protected:
    char* ptr;
    size_t byteSize;
    bool shared;
    MemoryKind kind;
    Ref<DeviceT> device;
    BufferAllocatorT allocator;
  };

  // Memory object backed by a buffer
  class Memory
  {
  public:
    Memory() : bufferOffset(0) {}
    virtual ~Memory() = default;

    Memory(const Ref<Buffer>& buffer, size_t bufferOffset = 0)
      : buffer(buffer),
        bufferOffset(bufferOffset) {}

    Buffer* getBuffer() const { return buffer.get(); }
    size_t getBufferOffset() const { return bufferOffset; }

    // If the buffer gets reallocated, this must be called to update the internal pointer
    virtual void updatePtr() = 0;

  protected:
    Ref<Buffer> buffer;  // buffer containing the data
    size_t bufferOffset; // offset in the buffer
  };

} // namespace oidn
