// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "heap.h"
#include "buffer.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // Heap
  // -----------------------------------------------------------------------------------------------

  void Heap::attach(Buffer* buffer)
  {
    buffers.insert(buffer);
  }

  void Heap::detach(Buffer* buffer)
  {
    buffers.erase(buffer);
  }

  void Heap::preRealloc()
  {
    for (auto buffer : buffers)
      buffer->preRealloc();
  }

  void Heap::postRealloc()
  {
    for (auto buffer : buffers)
      buffer->postRealloc();
  }

  // -----------------------------------------------------------------------------------------------
  // USMHeap
  // -----------------------------------------------------------------------------------------------

  USMHeap::USMHeap(const Ref<Engine>& engine, size_t byteSize, Storage storage)
    : ptr(nullptr),
      byteSize(byteSize),
      storage(storage),
      engine(engine)
  {
    if (storage == Storage::Undefined)
      this->storage = Storage::Device;

    ptr = static_cast<char*>(engine->usmAlloc(byteSize, this->storage));
  }

  USMHeap::~USMHeap()
  {
    try
    {
      engine->usmFree(ptr, storage);
    }
    catch (...) {}
  }

  Engine* USMHeap::getEngine() const
  {
    return engine.get();
  }

  void USMHeap::realloc(size_t newByteSize)
  {
    if (newByteSize == byteSize)
      return;

    preRealloc();

    engine->usmFree(ptr, storage);
    ptr = static_cast<char*>(engine->usmAlloc(newByteSize, storage));
    byteSize = newByteSize;

    postRealloc();
  }

OIDN_NAMESPACE_END