// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "buffer.h"
#include "heap.h"
#include "arena.h"
#include "tensor.h"
#include "image.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // Buffer
  // -----------------------------------------------------------------------------------------------

  Buffer::Buffer(const Ref<Arena>& arena, size_t byteOffset)
    : arena(arena),
      byteOffset(byteOffset)
  {
    arena->getHeap()->attach(this);
  }

  Buffer::~Buffer()
  {
    if (arena)
      arena->getHeap()->detach(this);
  }

  Device* Buffer::getDevice() const
  {
    return getEngine()->getDevice();
  }

  void Buffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    throw Exception(Error::InvalidOperation, "reading the buffer is not supported");
  }

  void Buffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    throw Exception(Error::InvalidOperation, "writing the buffer is not supported");
  }

  Ref<Buffer> Buffer::newBuffer(size_t byteSize, size_t byteOffset)
  {
    if (!arena)
      throw Exception(Error::InvalidOperation, "cannot suballocate a buffer without an arena");
    if (byteOffset + byteSize > getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");

    return arena->newBuffer(byteSize, this->byteOffset + byteOffset);
  }

  Ref<Tensor> Buffer::newTensor(const TensorDesc& desc, size_t byteOffset)
  {
    return getEngine()->newTensor(this, desc, byteOffset);
  }

  Ref<Image> Buffer::newImage(const ImageDesc& desc, size_t byteOffset)
  {
    return makeRef<Image>(this, desc, byteOffset);
  }

  Buffer* Buffer::toUser()
  {
    // User-owned buffers must hold a reference to the device to keep it alive
    userBufferDevice = getDevice();
    return this;
  }

  void Buffer::attach(Memory* mem)
  {
    mems.insert(mem);
  }

  void Buffer::detach(Memory* mem)
  {
    mems.erase(mem);
  }

  void Buffer::preRealloc()
  {
    for (Memory* mem : mems)
      mem->preRealloc();
  }

  void Buffer::postRealloc()
  {
    for (Memory* mem : mems)
      mem->postRealloc();
  }

  // -----------------------------------------------------------------------------------------------
  // USMBuffer
  // -----------------------------------------------------------------------------------------------

  USMBuffer::USMBuffer(Engine* engine)
    : engine(engine),
      ptr(nullptr),
      byteSize(0),
      shared(true),
      storage(Storage::Undefined)
  {}

  USMBuffer::USMBuffer(Engine* engine, size_t byteSize, Storage storage)
    : engine(engine),
      ptr(nullptr),
      byteSize(byteSize),
      shared(false),
      storage(storage)
  {
    if (storage == Storage::Undefined)
      this->storage = getDevice()->isManagedMemorySupported() ? Storage::Managed : Storage::Host;

    ptr = static_cast<char*>(engine->usmAlloc(byteSize, this->storage));
  }

  USMBuffer::USMBuffer(Engine* engine, void* data, size_t byteSize, Storage storage)
    : engine(engine),
      ptr(static_cast<char*>(data)),
      byteSize(byteSize),
      shared(true),
      storage(storage)
  {
    if (ptr == nullptr)
      throw Exception(Error::InvalidArgument, "buffer pointer is null");

    if (storage == Storage::Undefined)
      this->storage = engine->getDevice()->getPtrStorage(ptr);
  }

  USMBuffer::USMBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
    : Buffer(arena, byteOffset),
      engine(arena->getHeap()->getEngine()),
      ptr(nullptr),
      byteSize(byteSize),
      shared(true),
      storage(arena->getHeap()->getStorage())
  {
    if (byteOffset + byteSize > arena->getByteSize())
      throw Exception(Error::InvalidArgument, "arena region is out of bounds");

    USMHeap* heap = dynamic_cast<USMHeap*>(arena->getHeap());
    if (!heap)
      throw Exception(Error::InvalidArgument, "buffer is incompatible with arena");

    ptr = heap->ptr + byteOffset;
  }

  USMBuffer::~USMBuffer()
  {
    if (!shared && ptr)
    {
      try
      {
        engine->usmFree(ptr, storage);
      }
      catch (...) {}
    }
  }

  void USMBuffer::postRealloc()
  {
    if (arena)
    {
      USMHeap* heap = static_cast<USMHeap*>(arena->getHeap());
      ptr = heap->ptr + byteOffset;
    }

    Buffer::postRealloc();
  }

  void USMBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    if (sync == SyncMode::Blocking)
      engine->usmCopy(dstHostPtr, ptr + byteOffset, byteSize);
    else
      engine->submitUSMCopy(dstHostPtr, ptr + byteOffset, byteSize);
  }

  void USMBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    if (sync == SyncMode::Blocking)
      engine->usmCopy(ptr + byteOffset, srcHostPtr, byteSize);
    else
      engine->submitUSMCopy(ptr + byteOffset, srcHostPtr, byteSize);
  }

OIDN_NAMESPACE_END
