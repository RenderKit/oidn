// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/heap.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalHeap : public Heap
  {
    friend class MetalBuffer;

  public:
    MetalHeap(const Ref<MetalEngine>& engine, size_t byteSize, Storage storage);
    ~MetalHeap();

    Engine* getEngine() const override;
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    void realloc(size_t newByteSize) override;

  private:
    void init();
    void free();

    id<MTLHeap> heap;
    size_t byteSize;
    Storage storage;
    Ref<MetalEngine> engine;
  };

OIDN_NAMESPACE_END