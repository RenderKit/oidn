// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  // 3x3 convolution descriptor
  struct ConvDesc
  {
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;
    bool relu;
  };

  // 3x3 convolution
  class Conv : public virtual Op
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;

  public:
    Conv(const ConvDesc& desc)
      : src(desc.src),
        weight(desc.weight),
        bias(desc.bias),
        dst(desc.dst) {}

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
