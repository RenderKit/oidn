// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"
#include "tensor.h"
#include "image.h"
#include <unordered_set>

OIDN_NAMESPACE_BEGIN

  class ScratchBuffer;

  // Manages scratch buffers sharing the same memory
  class ScratchBufferManager final
  {
    friend class ScratchBuffer;

  public:
    ScratchBufferManager(const Ref<Engine>& engine);

  private:
    // Allocation consisting of a regular buffer and a set of scratch buffers sharing the same memory
    struct Alloc
    {
      Ref<Buffer> buffer;
      std::unordered_set<ScratchBuffer*> scratches;
    };

    // Scratch buffers must attach themselves
    Buffer* attach(ScratchBuffer* scratch);
    void detach(ScratchBuffer* scratch);

    // Updates the pointers of all attached memory objects
    void updatePtrs(Alloc& alloc);

    Ref<Engine> engine;
    std::unordered_map<std::string, Alloc> allocs; // allocations by ID
  };

  // Scratch buffer that shares memory with other scratch buffers having the same ID
  class ScratchBuffer final : public Buffer
  {
    friend class ScratchBufferManager;

  public:
    ScratchBuffer(const std::shared_ptr<ScratchBufferManager>& manager, size_t byteSize,
                  const std::string& id);
    ~ScratchBuffer();

    Engine* getEngine() const override { return manager->engine.get(); }

    void* getPtr() const override { return parentBuffer->getPtr(); }
    void* getHostPtr() const override { return parentBuffer->getHostPtr(); }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return parentBuffer->getStorage(); }

    Buffer* getParentBuffer() const { return parentBuffer; }

  private:
    void attach(Memory* mem) override;
    void detach(Memory* mem) override;

    std::shared_ptr<ScratchBufferManager> manager;
    Buffer* parentBuffer; // buffer that backs the memory of this scratch buffer
    std::unordered_set<Memory*> mems; // attached memory objects
    size_t byteSize; // size of this buffer
    std::string id;
  };

OIDN_NAMESPACE_END
