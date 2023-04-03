// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"
#include "scratch.h"
#include "concat_conv_chw.h"
#include "concat_conv_hwc.h"

OIDN_NAMESPACE_BEGIN

  Ref<Buffer> Engine::newBuffer(size_t byteSize, Storage storage)
  {
    return makeRef<USMBuffer>(this, byteSize, storage);
  }

  Ref<Buffer> Engine::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<USMBuffer>(this, ptr, byteSize);
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

  Ref<ScratchBuffer> Engine::newScratchBuffer(size_t byteSize)
  {
    auto scratchManager = scratchManagerWp.lock();
    if (!scratchManager)
      scratchManagerWp = scratchManager = std::make_shared<ScratchBufferManager>(this);
    return makeRef<ScratchBuffer>(scratchManager, byteSize);
  }

  std::shared_ptr<Tensor> Engine::newTensor(const TensorDesc& desc, Storage storage)
  {
    return std::make_shared<GenericTensor>(this, desc, storage);
  }

  std::shared_ptr<Tensor> Engine::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
  {
    assert(buffer->getEngine() == this);
    return std::make_shared<GenericTensor>(buffer, desc, byteOffset);
  }

  bool Engine::isConvSupported(PostOp postOp)
  {
    return postOp == PostOp::None;
  }

OIDN_NAMESPACE_END
