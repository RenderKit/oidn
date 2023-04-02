// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "buffer.h"
#include "tensor.h"
#include "image.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // Buffer
  // -----------------------------------------------------------------------------------------------

  void* Buffer::map(size_t byteOffset, size_t byteSize, Access access)
  {
    throw Exception(Error::InvalidOperation, "mapping the buffer is not supported");
  }

  void Buffer::unmap(void* hostPtr)
  {
    throw Exception(Error::InvalidOperation, "unmapping the buffer is not supported");
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
  // MappedBuffer
  // -----------------------------------------------------------------------------------------------

  MappedBuffer::MappedBuffer(const Ref<Buffer>& buffer, size_t byteOffset, size_t byteSize, Access access)
    : ptr((char*)buffer->map(byteOffset, byteSize, access)),
      byteSize(byteSize),
      buffer(buffer) {}

  MappedBuffer::~MappedBuffer()
  {
    buffer->unmap(ptr);
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
    ptr = (char*)engine->malloc(byteSize, storage);
  }

  USMBuffer::USMBuffer(const Ref<Engine>& engine, void* data, size_t byteSize, Storage storage)
    : ptr((char*)data),
      byteSize(byteSize),
      shared(true),
      storage(storage),
      engine(engine)
  {
    if (ptr == nullptr)
      throw Exception(Error::InvalidArgument, "buffer pointer null");
    if (storage == Storage::Undefined)
      storage = engine->getDevice()->getPointerStorage(ptr);
  }

  USMBuffer::~USMBuffer()
  {
    // Unmap all mapped regions
    unmapAll();

    // Free the memory
    if (!shared && ptr)
      engine->free(ptr, storage);
  }

  void* USMBuffer::map(size_t byteOffset, size_t byteSize, Access access)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    if (byteSize == 0)
      byteSize = this->byteSize - byteOffset; // entire buffer

    void* devPtr = ptr + byteOffset;
    if (storage != Storage::Device)
      return devPtr;

    void* hostPtr = engine->malloc(byteSize, Storage::Host);
    if (access != Access::WriteDiscard)
      engine->memcpy(hostPtr, devPtr, byteSize);
    
    mappedRegions.insert({hostPtr, {devPtr, byteSize, access}});
    return hostPtr;
  }

  void USMBuffer::unmap(void* hostPtr)
  {
    if (storage != Storage::Device)
      return;

    auto region = mappedRegions.find(hostPtr);
    if (region == mappedRegions.end())
      throw Exception(Error::InvalidArgument, "invalid mapped region");

    if (region->second.access != Access::Read)
      engine->memcpy(region->second.devPtr, hostPtr, byteSize);
    engine->free(hostPtr, Storage::Host);

    mappedRegions.erase(region);
  }

  void USMBuffer::unmapAll()
  {
    for (const auto& region : mappedRegions)
      unmap(region.first);
  }

  void USMBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (!mappedRegions.empty())
      throw Exception(Error::InvalidOperation, "buffer cannot be read while mapped");
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    if (sync == SyncMode::Sync)
      engine->memcpy(dstHostPtr, ptr + byteOffset, byteSize);
    else
      engine->submitMemcpy(dstHostPtr, ptr + byteOffset, byteSize);
  }

  void USMBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (!mappedRegions.empty())
      throw Exception(Error::InvalidOperation, "buffer cannot be written while mapped");
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    if (sync == SyncMode::Sync)
      engine->memcpy(ptr + byteOffset, srcHostPtr, byteSize);
    else
      engine->submitMemcpy(ptr + byteOffset, srcHostPtr, byteSize);
  }

  void USMBuffer::realloc(size_t newByteSize)
  {
    if (shared)
      throw std::logic_error("shared buffers cannot be reallocated");
    if (!mappedRegions.empty())
      throw std::logic_error("mapped buffers cannot be reallocated");

    engine->free(ptr, storage);
    ptr = (char*)engine->malloc(newByteSize, storage);
    byteSize = newByteSize;
  }

OIDN_NAMESPACE_END
