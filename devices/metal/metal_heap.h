// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/heap.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  class MetalHeap : public Heap
  {
    friend class MetalBuffer;

  public:
    MetalHeap(MetalEngine* engine, size_t byteSize, Storage storage);
    ~MetalHeap();

    Engine* getEngine() const override { return engine; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    void realloc(size_t newByteSize) override;

  private:
    void init();
    void free();

    MetalEngine* engine;
    id<MTLHeap> heap;
    size_t byteSize;
    Storage storage;
  };

OIDN_NAMESPACE_END