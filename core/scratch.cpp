// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scratch.h"

OIDN_NAMESPACE_BEGIN

  ScratchBufferManager::ScratchBufferManager(const Ref<Engine>& engine)
    : buffer(engine->newBuffer(0, Storage::Device))
  {
  }

  void ScratchBufferManager::attach(ScratchBuffer* scratch)
  {
    scratches.insert(scratch);

    if (scratch->localSize > buffer->getByteSize())
    {
      buffer->realloc(scratch->localSize);
      updatePtrs();
    }
  }

  void ScratchBufferManager::detach(ScratchBuffer* scratch)
  {
    scratches.erase(scratch);

    if (scratch->localSize == buffer->getByteSize())
    {
      size_t newGlobalSize = 0;
      for (auto scratch : scratches)
        newGlobalSize = max(newGlobalSize, scratch->getByteSize());

      if (newGlobalSize < buffer->getByteSize())
      {
        buffer->realloc(newGlobalSize);
        updatePtrs();
      }
    }
  }

  void ScratchBufferManager::updatePtrs()
  {
    for (auto scratch : scratches)
      for (auto mem : scratch->mems)
        mem->updatePtr();
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

  void ScratchBuffer::attach(Memory* mem)
  {
    mems.insert(mem);
  }

  void ScratchBuffer::detach(Memory* mem)
  {
    mems.erase(mem);
  }

OIDN_NAMESPACE_END