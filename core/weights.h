// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <unordered_set>
#include "data.h"
#include "tensor.h"

namespace oidn {

  // Parses, reorders and caches network weights and biases from a blob
  class Weights
  {
  private:
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    std::unordered_set<std::string> reordered;
    Ref<Device> device;

  public:
    Weights(const Ref<Device>& device, const Data& blob);
    const std::shared_ptr<Tensor>& get(const std::string& name);

  private:
    std::shared_ptr<Tensor> reorder1D(const std::shared_ptr<Tensor>& src);
    std::shared_ptr<Tensor> reorder4D(const std::shared_ptr<Tensor>& src);
  };

} // namespace oidn
