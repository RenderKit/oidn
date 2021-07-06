// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scratch.h"

namespace oidn {

  ScratchBufferManager::ScratchBufferManager(const Ref<Device>& device)
    : buffer(device->newBuffer(0, Buffer::Kind::Device))
  {
  }

  void ScratchBufferManager::attach(ScratchBuffer* scratch)
  {
    scratches.insert(scratch);

    if (scratch->localSize > buffer->size())
    {
      buffer->resize(scratch->localSize);
      updatePtrs();
    }
  }

  void ScratchBufferManager::detach(ScratchBuffer* scratch)
  {
    scratches.erase(scratch);

    if (scratch->localSize == buffer->size())
    {
      size_t newGlobalSize = 0;
      for (auto scratch : scratches)
        newGlobalSize = max(newGlobalSize, scratch->size());

      if (newGlobalSize < buffer->size())
      {
        buffer->resize(newGlobalSize);
        updatePtrs();
      }
    }
  }

  // Updates the pointers of all allocated memory objects
  void ScratchBufferManager::updatePtrs()
  {
    for (auto scratch : scratches)
    {
      for (auto& memWp : scratch->memWps)
      {
        if (auto mem = memWp.lock())
          mem->updatePtr();
      }
    }
  }

  ScratchBuffer::ScratchBuffer(const std::shared_ptr<ScratchBufferManager>& manager, size_t size)
    : manager(manager),
      localSize(size)
  {
    manager->attach(this);
  }

  ScratchBuffer::~ScratchBuffer()
  {
    manager->detach(this);
  }

  std::shared_ptr<Tensor> ScratchBuffer::newTensor(const TensorDesc& desc, ptrdiff_t offset)
  {
    size_t absOffset = offset >= 0 ? offset : localSize + offset;
    auto result = std::make_shared<Tensor>(this, desc, absOffset);
    memWps.push_back(result);
    return result;
  }

  std::shared_ptr<Image> ScratchBuffer::newImage(const ImageDesc& desc, ptrdiff_t offset)
  {
    size_t absOffset = offset >= 0 ? offset : localSize + offset;
    auto result = std::make_shared<Image>(this, desc, absOffset);
    memWps.push_back(result);
    return result;
  }

} // namespace oidn