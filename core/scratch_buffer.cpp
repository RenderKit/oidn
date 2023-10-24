// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scratch_buffer.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  ScratchBufferManager::ScratchBufferManager(const Ref<Engine>& engine)
    : engine(engine)
  {}

  // Attaches a scratch buffer and returns the parent buffer that backs its memory
  Buffer* ScratchBufferManager::attach(ScratchBuffer* scratch)
  {
    Alloc& alloc = allocs[scratch->id];

    if (alloc.buffer)
    {
      // Increase the size of the allocation if necessary
      if (scratch->byteSize > alloc.buffer->getByteSize())
      {
        alloc.buffer->realloc(scratch->byteSize);
        updatePtrs(alloc);
      }
    }
    else
    {
      // Allocate a new buffer
      alloc.buffer = engine->newBuffer(scratch->byteSize, Storage::Device);
    }

    alloc.scratches.insert(scratch);
    return alloc.buffer.get();
  }

  // Detaches the scratch buffer
  void ScratchBufferManager::detach(ScratchBuffer* scratch)
  {
    Alloc& alloc = allocs[scratch->id];

    alloc.scratches.erase(scratch);

    if (alloc.scratches.empty())
    {
      // Free the allocation because no more scratch buffers are attached
      allocs.erase(scratch->id);
    }
    else
    {
      // Shrink the size of the allocation if possible
      if (scratch->byteSize == alloc.buffer->getByteSize())
      {
        size_t newBufferByteSize = 0;
        for (auto otherScratch : alloc.scratches)
          newBufferByteSize = max(newBufferByteSize, otherScratch->byteSize);

        if (newBufferByteSize < alloc.buffer->getByteSize())
        {
          alloc.buffer->realloc(newBufferByteSize);
          updatePtrs(alloc);
        }
      }
    }
  }

  void ScratchBufferManager::updatePtrs(Alloc& alloc)
  {
    for (auto scratch : alloc.scratches)
      for (auto mem : scratch->mems)
        mem->updatePtr();
  }

  ScratchBuffer::ScratchBuffer(const std::shared_ptr<ScratchBufferManager>& manager, size_t byteSize,
                               const std::string& id)
    : manager(manager),
      parentBuffer(nullptr),
      byteSize(byteSize),
      id(id)
  {
    parentBuffer = manager->attach(this);
  }

  ScratchBuffer::~ScratchBuffer()
  {
    try
    {
      manager->detach(this);
    }
    catch (...) {}
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