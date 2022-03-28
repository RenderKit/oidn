// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "buffer.h"
#include "tensor.h"
#include "image.h"

namespace oidn {

  void* Buffer::map(size_t byteOffset, size_t byteSize, Access access)
  {
    throw Exception(Error::InvalidOperation, "mapping the buffer is not supported");
  }

  void Buffer::unmap(void* hostPtr)
  {
    throw Exception(Error::InvalidOperation, "unmapping the buffer is not supported");
  }

  void Buffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr)
  {
    throw Exception(Error::InvalidOperation, "reading the buffer is not supported");
  }

  void Buffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    throw Exception(Error::InvalidOperation, "writing the buffer is not supported");
  }

  void Buffer::realloc(size_t newByteSize)
  {
    throw std::logic_error("reallocating the buffer is not supported");
  }

  std::shared_ptr<Tensor> Buffer::newTensor(const TensorDesc& desc, ptrdiff_t relByteOffset)
  {
    size_t byteOffset = relByteOffset >= 0 ? relByteOffset : getByteSize() + relByteOffset;
    return getDevice()->newTensor(this, desc, byteOffset);
  }

  std::shared_ptr<Image> Buffer::newImage(const ImageDesc& desc, ptrdiff_t relByteOffset)
  {
    size_t byteOffset = relByteOffset >= 0 ? relByteOffset : getByteSize() + relByteOffset;
    return std::make_shared<Image>(this, desc, byteOffset);
  }

  MappedBuffer::MappedBuffer(const Ref<Buffer>& buffer, size_t byteOffset, size_t byteSize, Access access)
    : ptr((char*)buffer->map(byteOffset, byteSize, access)),
      byteSize(byteSize),
      buffer(buffer) {}

  MappedBuffer::~MappedBuffer()
  {
    buffer->unmap(ptr);
  }

  USMBuffer::USMBuffer(const Ref<Device>& device, size_t byteSize, Storage storage)
    : ptr(nullptr),
      byteSize(byteSize),
      shared(false),
      storage(storage),
      device(device)
  {
    ptr = (char*)device->malloc(byteSize, storage);
  }

  USMBuffer::USMBuffer(const Ref<Device>& device, void* data, size_t byteSize)
    : ptr((char*)data),
      byteSize(byteSize),
      shared(true),
      storage(Storage::Undefined),
      device(device)
  {
    if (ptr == nullptr)
      throw Exception(Error::InvalidArgument, "buffer pointer null");
  }

  USMBuffer::~USMBuffer()
  {
    if (shared)
      return;

    for (const auto& region : mappedRegions)
      unmap(region.first);

    device->free(ptr, storage);
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

    void* hostPtr = device->malloc(byteSize, Storage::Host);
    if (access != Access::WriteDiscard)
      device->memcpy(hostPtr, devPtr, byteSize);
    
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
      device->memcpy(region->second.devPtr, hostPtr, byteSize);
    device->free(hostPtr, Storage::Host);

    mappedRegions.erase(region);
  }

  void USMBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr)
  {
    if (!mappedRegions.empty())
      throw Exception(Error::InvalidOperation, "buffer cannot be read while mapped");
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    device->memcpy(dstHostPtr, ptr + byteOffset, byteSize);
  }

  void USMBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr)
  {
    if (!mappedRegions.empty())
      throw Exception(Error::InvalidOperation, "buffer cannot be written while mapped");
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    device->memcpy(ptr + byteOffset, srcHostPtr, byteSize);
  }

  void USMBuffer::realloc(size_t newByteSize)
  {
    if (shared)
      throw std::logic_error("shared buffers cannot be reallocated");
    if (!mappedRegions.empty())
      throw std::logic_error("mapped buffers cannot be reallocated");

    device->free(ptr, storage);
    ptr = (char*)device->malloc(newByteSize, storage);
    byteSize = newByteSize;
  }

} // namespace oidn
