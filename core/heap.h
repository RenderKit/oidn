// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "ref.h"
#include <unordered_set>

OIDN_NAMESPACE_BEGIN

  class Engine;
  class Buffer;

  // -----------------------------------------------------------------------------------------------
  // Heap
  // -----------------------------------------------------------------------------------------------

  class Heap : public RefCount
  {
    friend class Buffer;

  public:
    virtual Engine* getEngine() const = 0;
    virtual size_t getByteSize() const = 0;
    virtual Storage getStorage() const = 0;

    virtual void realloc(size_t newByteSize) = 0;

  protected:
    void preRealloc();
    void postRealloc();

  private:
    void attach(Buffer* buffer);
    void detach(Buffer* buffer);

    std::unordered_set<Buffer*> buffers;
  };

  // -----------------------------------------------------------------------------------------------
  // USMHeap
  // -----------------------------------------------------------------------------------------------

  class USMBuffer;

  // Unified shared memory (USM) based heap
  class USMHeap : public Heap
  {
    friend class USMBuffer;

  public:
    USMHeap(Engine* engine, size_t byteSize, Storage storage);
    ~USMHeap();

    Engine* getEngine() const override { return engine; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    void realloc(size_t newByteSize) override;

  private:
    Engine* engine;
    char* ptr;
    size_t byteSize;
    Storage storage;
  };

OIDN_NAMESPACE_END