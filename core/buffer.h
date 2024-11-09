// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "ref.h"
#include <unordered_set>

OIDN_NAMESPACE_BEGIN

  struct TensorDesc;
  struct ImageDesc;

  class Device;
  class Engine;
  class Heap;
  class Arena;
  class Memory;
  class Tensor;
  class Image;

  // -----------------------------------------------------------------------------------------------
  // Buffer
  // -----------------------------------------------------------------------------------------------

  // Generic buffer object
  class Buffer : public RefCount
  {
    friend class Heap;
    friend class Memory;

  public:
    Buffer() = default;
    Buffer(const Ref<Arena>& arena, size_t byteOffset);
    ~Buffer();

    virtual Engine* getEngine() const = 0;
    Device* getDevice() const;
    virtual void* getPtr() const = 0;     // pointer in device address space
    virtual void* getHostPtr() const = 0; // pointer in host address space if available, nullptr otherwise
    virtual size_t getByteSize() const = 0;
    virtual bool isShared() const = 0;
    virtual Storage getStorage() const = 0;
    Arena* getArena() const { return arena.get(); }

    virtual void read(size_t byteOffset, size_t byteSize, void* dstHostPtr,
                      SyncMode sync = SyncMode::Blocking);

    virtual void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr,
                       SyncMode sync = SyncMode::Blocking);

    Ref<Buffer> newBuffer(size_t byteSize, size_t byteOffset = 0);
    Ref<Tensor> newTensor(const TensorDesc& desc, size_t byteOffset = 0);
    Ref<Image> newImage(const ImageDesc& desc, size_t byteOffset = 0);

    // Before returning a buffer to the user, it must be converted to a user buffer
    Buffer* toUser();

  private:
    Ref<Device> userBufferDevice; // user buffers must keep a reference to the device

  protected:
    virtual void preRealloc();
    virtual void postRealloc();

    Ref<Arena> arena;      // arena where the buffer is allocated (optional)
    size_t byteOffset = 0; // offset of the buffer in the arena

  private:
    // Memory objects backed by the buffer must attach themselves
    void attach(Memory* mem);
    void detach(Memory* mem);

    std::unordered_set<Memory*> mems;
  };

  // -----------------------------------------------------------------------------------------------
  // USMBuffer
  // -----------------------------------------------------------------------------------------------

  // Unified shared memory (USM) based buffer object
  class USMBuffer : public Buffer
  {
    friend class USMHeap;

  public:
    USMBuffer(Engine* engine, size_t byteSize, Storage storage);
    USMBuffer(Engine* engine, void* data, size_t byteSize, Storage storage = Storage::Undefined);
    USMBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset);
    ~USMBuffer();

    Engine* getEngine() const override { return engine; }

    void* getPtr() const override { return ptr; }
    void* getHostPtr() const override { return ptr; }
    size_t getByteSize() const override { return byteSize; }
    bool isShared() const override { return shared; }
    Storage getStorage() const override { return storage; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync) override;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync) override;

  protected:
    explicit USMBuffer(Engine* engine);

    void postRealloc() override;

    Engine* engine;
    char* ptr;
    size_t byteSize;
    bool shared;
    Storage storage;
  };

  // -----------------------------------------------------------------------------------------------
  // Memory
  // -----------------------------------------------------------------------------------------------

  // Memory object optionally backed by a buffer
  class Memory : public RefCount
  {
    friend class Buffer;

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

  protected:
    // If the buffer gets reallocated, these must be called to free/re-init internal resources
    virtual void preRealloc() {}
    virtual void postRealloc() {}

    Ref<Buffer> buffer; // buffer containing the data (optional)
    size_t byteOffset;  // offset of the data in the buffer
  };

OIDN_NAMESPACE_END
