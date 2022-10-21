// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"
#include <unordered_map>

namespace oidn {

  struct TensorDesc;
  struct ImageDesc;

  class Memory;
  class Tensor;
  class Image;

  // ---------------------------------------------------------------------------
  // Buffer
  // ---------------------------------------------------------------------------

  // Generic buffer object
  class Buffer : public RefCount
  {
    friend class Memory;

  public:
    virtual Engine* getEngine() const = 0;
    Device* getDevice() const { return getEngine()->getDevice(); }

    virtual char* getData() = 0;
    virtual const char* getData() const = 0;
    virtual size_t getByteSize() const = 0;
    virtual Storage getStorage() const = 0;

    virtual void* map(size_t byteOffset, size_t byteSize, Access access);
    virtual void unmap(void* hostPtr);

    virtual void read(size_t byteOffset, size_t byteSize, void* dstHostPtr);
    virtual void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr);

    // Reallocates the buffer with a new size discarding its current contents
    virtual void realloc(size_t newByteSize);

    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, ptrdiff_t relByteOffset);
    std::shared_ptr<Image> newImage(const ImageDesc& desc, ptrdiff_t relByteOffset);

  private:
    // Memory objects backed by the buffer must attach themselves
    virtual void attach(Memory* mem) {}
    virtual void detach(Memory* mem) {}
  };

  // ---------------------------------------------------------------------------
  // MappedBuffer
  // ---------------------------------------------------------------------------

  // Memory mapped version of a buffer
  class MappedBuffer final : public Buffer
  {
  public:
    MappedBuffer(const Ref<Buffer>& buffer, size_t byteOffset, size_t byteSize, Access access);
    ~MappedBuffer();

    Engine* getEngine() const override { return buffer->getEngine(); }

    char* getData() override { return ptr; }
    const char* getData() const override { return ptr; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return Storage::Host; }

  private:
    char* ptr;
    size_t byteSize;
    Ref<Buffer> buffer;
  };

  // ---------------------------------------------------------------------------
  // USMBuffer
  // ---------------------------------------------------------------------------

  // Unified shared memory based buffer object
  class USMBuffer final : public Buffer
  {
  public:
    USMBuffer(const Ref<Engine>& engine, size_t byteSize, Storage storage);
    USMBuffer(const Ref<Engine>& engine, void* data, size_t byteSize);
    ~USMBuffer();

    Engine* getEngine() const override { return engine.get(); }

    char* getData() override { return ptr; }
    const char* getData() const override { return ptr; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    void* map(size_t byteOffset, size_t byteSize, Access access) override;
    void unmap(void* hostPtr) override;

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr) override;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr) override;

    void realloc(size_t newByteSize) override;

  private:
    struct MappedRegion
    {
      void* devPtr;
      size_t byteSize;
      Access access;
    };

    char* ptr;
    size_t byteSize;
    bool shared;
    Storage storage;
    std::unordered_map<void*, MappedRegion> mappedRegions;
    Ref<Engine> engine;
  };

  // ---------------------------------------------------------------------------
  // Memory
  // ---------------------------------------------------------------------------

  // Memory object optionally backed by a buffer
  class Memory
  {
  public:
    Memory() : byteOffset(0) {}

    Memory(const Ref<Buffer>& buffer, size_t byteOffset = 0)
      : buffer(buffer),
        byteOffset(byteOffset)
    {
      buffer->attach(this);
    }

    virtual ~Memory()
    {
      if (buffer)
        buffer->detach(this);
    }

    Buffer* getBuffer() const { return buffer.get(); }
    size_t getByteOffset() const { return byteOffset; }

    // If the buffer gets reallocated, this must be called to update the internal pointer
    virtual void updatePtr() = 0;

  protected:
    Ref<Buffer> buffer; // buffer containing the data
    size_t byteOffset;  // offset in the buffer
  };

} // namespace oidn
