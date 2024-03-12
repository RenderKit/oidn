// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "arena.h"
#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  // Subdevice consisting of an engine and some shared resources
  class Subdevice final
  {
  public:
    explicit Subdevice(std::unique_ptr<Engine>&& engine);

    Engine* getEngine() const { return engine.get(); }

    // Scratch
    Ref<Arena> newScratchArena(size_t byteSize, const std::string& name = "");
    void trimScratch();

    // Tensor cache
    std::shared_ptr<TensorMap> getCachedTensors(const void* key);

  private:
    // Disable copying
    Subdevice(const Subdevice&) = delete;
    Subdevice& operator =(const Subdevice&) = delete;

    std::unique_ptr<Engine> engine; // must be declared first / destroyed last

    // Resources
    std::unique_ptr<ScratchArenaManager> scratchArenaManager;
    std::unordered_map<const void*, std::shared_ptr<TensorMap>> cachedTensors; // cached weights
  };

OIDN_NAMESPACE_END