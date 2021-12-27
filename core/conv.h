// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  // 3x3 convolution descriptor
  struct ConvDesc
  {
    std::string name;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;
    bool relu;
  };

  // 3x3 convolution node
  class ConvNode : public virtual Node
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> weights;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;

  public:
    ConvNode(const ConvDesc& desc)
      : src(desc.src),
        weights(desc.weights),
        bias(desc.bias),
        dst(desc.dst) {}

    std::shared_ptr<Tensor> getDst() const { return dst; }
  };

} // namespace oidn
