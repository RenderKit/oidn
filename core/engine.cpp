// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"
#include "conv.h"

OIDN_NAMESPACE_BEGIN

  void Engine::setSubdevice(Subdevice* subdevice)
  {
    if (this->subdevice)
      throw std::logic_error("subdevice already set");
    this->subdevice = subdevice;
  }

  Ref<Heap> Engine::newHeap(size_t byteSize, Storage storage)
  {
    return makeRef<USMHeap>(this, byteSize, storage);
  }

  // Returns the actual size and required alignment of a buffer allocated from a heap/arena
  SizeAndAlignment Engine::getBufferByteSizeAndAlignment(size_t byteSize, Storage storage)
  {
    return {byteSize, memoryAlignment};
  }

  Ref<Buffer> Engine::newBuffer(size_t byteSize, Storage storage)
  {
    return makeRef<USMBuffer>(this, byteSize, storage);
  }

  Ref<Buffer> Engine::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<USMBuffer>(this, ptr, byteSize);
  }

  Ref<Buffer> Engine::newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
  {
    return makeRef<USMBuffer>(arena, byteSize, byteOffset);
  }

  Ref<Buffer> Engine::newNativeBuffer(void* handle)
  {
    throw Exception(Error::InvalidOperation,
      "creating a shared buffer from a native handle is not supported by the device");
  }

  Ref<Buffer> Engine::newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                        int fd, size_t byteSize)
  {
    throw Exception(Error::InvalidOperation,
      "creating a shared buffer from a POSIX file descriptor is not supported by the device");
  }

  Ref<Buffer> Engine::newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                        void* handle, const void* name, size_t byteSize)
  {
    throw Exception(Error::InvalidOperation,
      "creating a shared buffer from a Win32 handle is not supported by the device");
  }

  bool Engine::isSupported(const TensorDesc& desc) const
  {
    // We store tensor byte offsets in 32-bit unsigned integers
    return desc.getByteSize() <= UINT32_MAX;
  }

  Ref<Tensor> Engine::newTensor(const TensorDesc& desc, Storage storage)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");

    return makeRef<DeviceTensor>(this, desc, storage);
  }

  Ref<Tensor> Engine::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");
    if (buffer->getEngine() != this)
      throw std::invalid_argument("buffer was created by a different engine");

    return makeRef<DeviceTensor>(buffer, desc, byteOffset);
  }

  bool Engine::isConvSupported(PostOp postOp)
  {
    return postOp == PostOp::None;
  }

  void* Engine::usmAlloc(size_t byteSize, Storage storage)
  {
    throw std::logic_error("USM is not supported by the device");
  }

  void Engine::usmFree(void* ptr, Storage storage)
  {
    throw std::logic_error("USM is not supported by the device");
  }

  void Engine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    throw std::logic_error("USM is not supported by the device");
  }

  void Engine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    throw std::logic_error("USM is not supported by the device");
  }

  int Engine::getMaxWorkGroupSize() const
  {
    throw std::logic_error("getting the maximum work-group size is not supported by the device");
  }

  int Engine::getSubgroupSize() const
  {
    throw std::logic_error("getting the subgroup size is not supported by the device");
  }

OIDN_NAMESPACE_END
