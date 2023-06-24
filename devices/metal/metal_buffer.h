// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/buffer.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalBuffer : public Buffer
  {
  public:
    MetalBuffer(const Ref<Engine>& engine, size_t byteSize, Storage storage);
    
    ~MetalBuffer();
    
    Engine* getEngine() const override { return engine.get(); }
    
    char* getData() override;
    const char* getData() const override;
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }
      
    MTLBuffer_t getMetalBuffer() {return buffer;}
    
    void* map(size_t byteOffset, size_t byteSize, Access access) override;
    void unmap(void* hostPtr) override;
    
    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync = SyncMode::Sync) override;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync = SyncMode::Sync) override;

    // Reallocates the buffer with a new size discarding its current contents
    void realloc(size_t newByteSize) override;
    
  private:
    void init();
    void free();
    
  private:
    Ref<Engine> engine;
    size_t byteSize;
    Storage storage;
    MTLBuffer_t buffer;
    MTLCommandQueue_t commandQueue;
    
    struct MappedRegion
    {
      void* devPtr;
      size_t byteOffset;
      size_t byteSize;
      Access access;
    };

    std::unordered_map<void*, MappedRegion> mappedRegions;
  };

OIDN_NAMESPACE_END
