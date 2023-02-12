// Copyright 2009-2022 Intel Corporation
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
    // Scratch buffers must attach themselves
    void attach(ScratchBuffer* scratch);
    void detach(ScratchBuffer* scratch);

    // Updates the pointers of all attached memory objects
    void updatePtrs();

    Ref<Buffer> buffer; // global shared buffer
    std::unordered_set<ScratchBuffer*> scratches; // attached scratch buffers
  };

  // Scratch buffer that shares memory with other scratch buffers
  class ScratchBuffer final : public Buffer
  {
    friend class ScratchBufferManager;

  public:
    ScratchBuffer(const std::shared_ptr<ScratchBufferManager>& manager, size_t size);
    ~ScratchBuffer();

    Engine* getEngine() const override { return manager->buffer->getEngine(); }

    char* getData() override { return manager->buffer->getData(); }
    const char* getData() const override { return manager->buffer->getData(); };
    size_t getByteSize() const override { return localSize; }
    Storage getStorage() const override { return manager->buffer->getStorage(); }

  private:
    void attach(Memory* mem) override;
    void detach(Memory* mem) override;

    std::shared_ptr<ScratchBufferManager> manager;
    std::unordered_set<Memory*> mems; // attached memory objects
    size_t localSize; // size of this buffer
  };

OIDN_NAMESPACE_END
