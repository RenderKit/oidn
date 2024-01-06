// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "arena.h"
#include "heap.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // ScratchArenaManager
  // -----------------------------------------------------------------------------------------------

  ScratchArenaManager::ScratchArenaManager(Engine* engine)
    : engine(engine)
  {}

  // Attaches a scratch arena and returns the heap that backs its memory
  Heap* ScratchArenaManager::attach(ScratchArena* arena)
  {
    Alloc& alloc = allocs[arena->name];

    if (alloc.heap)
    {
      // Increase the size of the heap if necessary
      if (arena->byteSize > alloc.heap->getByteSize())
        alloc.heap->realloc(arena->byteSize);
    }
    else
    {
      // Allocate a new heap
      alloc.heap = engine->newHeap(arena->byteSize, Storage::Device);
    }

    alloc.arenas.insert(arena);
    return alloc.heap.get();
  }

  // Detaches the scratch arena
  void ScratchArenaManager::detach(ScratchArena* arena)
  {
    Alloc& alloc = allocs[arena->name];

    alloc.arenas.erase(arena);

    if (alloc.arenas.empty())
    {
      // Free the heap because no more arenas are attached
      allocs.erase(arena->name);
    }
    else
    {
      // Shrink the size of the heap if possible
      if (arena->byteSize == alloc.heap->getByteSize())
      {
        size_t newHeapByteSize = 0;
        for (auto otherArena : alloc.arenas)
          newHeapByteSize = max(newHeapByteSize, otherArena->byteSize);

        if (newHeapByteSize < alloc.heap->getByteSize())
          alloc.heap->realloc(newHeapByteSize);
      }
    }
  }

  // -----------------------------------------------------------------------------------------------
  // ScratchArena
  // -----------------------------------------------------------------------------------------------

  ScratchArena::ScratchArena(const std::shared_ptr<ScratchArenaManager>& manager, size_t byteSize,
                             const std::string& name)
    : manager(manager),
      heap(nullptr),
      byteSize(byteSize),
      name(name)
  {
    heap = manager->attach(this);
  }

  ScratchArena::~ScratchArena()
  {
    try
    {
      manager->detach(this);
    }
    catch (...) {}
  }

  Storage ScratchArena::getStorage() const
  {
    return heap->getStorage();
  }

  Ref<Buffer> ScratchArena::newBuffer(size_t byteSize, size_t byteOffset)
  {
    return manager->engine->newBuffer(this, byteSize, byteOffset);
  }

OIDN_NAMESPACE_END