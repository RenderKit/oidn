// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "buffer.h"
#include "tensor.h"
#include "image.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // Buffer
  // -----------------------------------------------------------------------------------------------

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

  void Buffer::realloc(size_t newByteSize)
  {
    throw std::logic_error("reallocating the buffer is not supported");
  }

  std::shared_ptr<Tensor> Buffer::newTensor(const TensorDesc& desc, size_t byteOffset)
  {
    return getEngine()->newTensor(this, desc, byteOffset);
  }

  std::shared_ptr<Image> Buffer::newImage(const ImageDesc& desc, size_t byteOffset)
  {
    return std::make_shared<Image>(this, desc, byteOffset);
  }

  // -----------------------------------------------------------------------------------------------
  // USMBuffer
  // -----------------------------------------------------------------------------------------------

  USMBuffer::USMBuffer(const Ref<Engine>& engine)
    : ptr(nullptr),
      byteSize(0),
      shared(true),
      storage(Storage::Undefined),
      engine(engine)
  {}

  USMBuffer::USMBuffer(const Ref<Engine>& engine, size_t byteSize, Storage storage)
    : ptr(nullptr),
      byteSize(byteSize),
      shared(false),
      storage(storage),
      engine(engine)
  {
    if (storage == Storage::Undefined)
      this->storage = getDevice()->isManagedMemorySupported() ? Storage::Managed : Storage::Host;

    ptr = static_cast<char*>(engine->usmAlloc(byteSize, this->storage));
  }

  USMBuffer::USMBuffer(const Ref<Engine>& engine, void* data, size_t byteSize, Storage storage)
    : ptr(static_cast<char*>(data)),
      byteSize(byteSize),
      shared(true),
      storage(storage),
      engine(engine)
  {
    if (ptr == nullptr)
      throw Exception(Error::InvalidArgument, "buffer pointer is null");

    if (storage == Storage::Undefined)
      this->storage = engine->getDevice()->getPointerStorage(ptr);
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

  void USMBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of range");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    if (sync == SyncMode::Sync)
      engine->usmCopy(dstHostPtr, ptr + byteOffset, byteSize);
    else
      engine->submitUSMCopy(dstHostPtr, ptr + byteOffset, byteSize);
  }

  void USMBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of range");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    if (sync == SyncMode::Sync)
      engine->usmCopy(ptr + byteOffset, srcHostPtr, byteSize);
    else
      engine->submitUSMCopy(ptr + byteOffset, srcHostPtr, byteSize);
  }

  void USMBuffer::realloc(size_t newByteSize)
  {
    if (shared)
      throw std::logic_error("shared buffers cannot be reallocated");

    engine->usmFree(ptr, storage);
    ptr = static_cast<char*>(engine->usmAlloc(newByteSize, storage));
    byteSize = newByteSize;
  }

OIDN_NAMESPACE_END
