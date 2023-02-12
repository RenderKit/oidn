// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include "data.h"
#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  // Parses, reorders and caches network weights and biases from a blob
  class Weights final
  {
  public:
    Weights(const Ref<Engine>& engine, const Data& blob);
    
    const std::shared_ptr<Tensor>& get(const std::string& name);

    // Returns the total amount of allocated scratch memory (for the reordered tensors)
    size_t getScratchByteSize() const;

  private:
    std::shared_ptr<Tensor> reorder1D(const std::shared_ptr<Tensor>& src);
    std::shared_ptr<Tensor> reorder4D(const std::shared_ptr<Tensor>& src);

    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    std::unordered_set<std::string> reordered;
    Ref<Engine> engine;
  };

OIDN_NAMESPACE_END
