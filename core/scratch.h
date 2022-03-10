// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "buffer.h"
#include "tensor.h"
#include "image.h"
#include <vector>
#include <unordered_set>

namespace oidn {

  class ScratchBuffer;

  // Manages scratch buffers sharing the same memory
  class ScratchBufferManager final
  {
    friend class ScratchBuffer;

  public:
    ScratchBufferManager(const Ref<Device>& device);

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

    Device* getDevice() override { return manager->buffer->getDevice(); }

    char* getData() override { return manager->buffer->getData(); }
    const char* getData() const override { return manager->buffer->getData(); };
    size_t getByteSize() const override { return localSize; }

    void* map(size_t offset, size_t size) override { return manager->buffer->map(offset, size); }
    void unmap(void* mappedPtr) override { return manager->buffer->unmap(mappedPtr); }

  private:
    void attach(Memory* mem) override;
    void detach(Memory* mem) override;

    std::shared_ptr<ScratchBufferManager> manager;
    std::unordered_set<Memory*> mems; // attached memory objects
    size_t localSize; // size of this buffer
  };

} // namespace oidn
