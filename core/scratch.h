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
  class ScratchBufferManager
  {
    friend class ScratchBuffer;

  private:
    Ref<Buffer> buffer;                           // global shared buffer
    std::unordered_set<ScratchBuffer*> scratches; // attached scratch buffers

  public:
    ScratchBufferManager(const Ref<Device>& device);

  private:
    void attach(ScratchBuffer* scratch);
    void detach(ScratchBuffer* scratch);
    void updatePtrs();
  };

  // Scratch buffer that shares memory with other scratch buffers
  class ScratchBuffer : public Buffer
  {
    friend class ScratchBufferManager;

  private:
    std::shared_ptr<ScratchBufferManager> manager;
    std::vector<std::weak_ptr<Memory>> memWps; // allocated memory objects
    size_t localSize;                          // size of this buffer

  public:
    ScratchBuffer(const std::shared_ptr<ScratchBufferManager>& manager, size_t size);
    ~ScratchBuffer();

    char* data() override { return manager->buffer->data(); }
    const char* data() const override { return manager->buffer->data(); };
    size_t size() const override { return localSize; }

    void* map(size_t offset, size_t size) override { return manager->buffer->map(offset, size); }
    void unmap(void* mappedPtr) override { return manager->buffer->unmap(mappedPtr); }

    Device* getDevice() override { return manager->buffer->getDevice(); }

    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, ptrdiff_t offset);
    std::shared_ptr<Image> newImage(const ImageDesc& desc, ptrdiff_t offset);
  };

} // namespace oidn
