// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "ref.h"
#include <unordered_set>
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  class Engine;
  class Heap;
  class Buffer;

  // -----------------------------------------------------------------------------------------------
  // Arena
  // -----------------------------------------------------------------------------------------------

  class Arena : public RefCount
  {
    friend class Buffer;

  public:
    virtual Engine* getEngine() const = 0;
    virtual Heap* getHeap() const = 0;
    virtual size_t getByteSize() const = 0;
    virtual Storage getStorage() const = 0;

    virtual Ref<Buffer> newBuffer(size_t byteSize, size_t byteOffset = 0) = 0;
  };

  // -----------------------------------------------------------------------------------------------
  // ScratchArenaManager
  // -----------------------------------------------------------------------------------------------

  class ScratchArena;

  // Manages scratch arenas sharing the same memory
  class ScratchArenaManager final
  {
    friend class ScratchArena;

  public:
    ScratchArenaManager(Engine* engine);

    // Trim the heap(s) to the minimum size required by the attached arenas
    void trim();

  private:
    // Allocation consisting of a heap and a set of scratch arenas sharing this heap
    struct Alloc
    {
      Ref<Heap> heap;
      std::unordered_set<ScratchArena*> arenas;
    };

    // Scratch arenas must attach themselves
    Heap* attach(ScratchArena* arena);
    void detach(ScratchArena* arena);

    Engine* engine;
    std::unordered_map<std::string, Alloc> allocs;
  };

  // -----------------------------------------------------------------------------------------------
  // ScratchArena
  // -----------------------------------------------------------------------------------------------

  // Scratch arena that shares memory with other scratch arenas having the same name
  class ScratchArena final : public Arena
  {
    friend class ScratchArenaManager;

  public:
    ScratchArena(ScratchArenaManager* manager, size_t byteSize,
                 const std::string& name);
    ~ScratchArena();

    Engine* getEngine() const override { return manager->engine; }
    Heap* getHeap() const override { return heap; }
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override;

    Ref<Buffer> newBuffer(size_t byteSize, size_t byteOffset) override;

  private:
    ScratchArenaManager* manager;
    Heap* heap;       // heap that backs the memory of this arena
    size_t byteSize;  // size of this arena
    std::string name;
  };

OIDN_NAMESPACE_END