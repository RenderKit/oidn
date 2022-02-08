// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  // Concatenation + 3x3 convolution descriptor
  struct ConcatConvDesc
  {
    std::shared_ptr<Tensor> src1;
    std::shared_ptr<Tensor> src2;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;
    bool relu;
  };

  // Concatenation + 3x3 convolution
  class ConcatConv : public virtual Op
  {
  protected:
    std::shared_ptr<Tensor> src1;
    std::shared_ptr<Tensor> src2;
    std::shared_ptr<Tensor> weight1;
    std::shared_ptr<Tensor> weight2;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;

  public:
    ConcatConv(const ConcatConvDesc& desc)
      : src1(desc.src1),
        src2(desc.src2),
        bias(desc.bias),
        dst(desc.dst) {}

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
