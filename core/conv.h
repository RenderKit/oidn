// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  // 3x3 convolution descriptor
  struct ConvDesc
  {
    TensorDesc srcDesc;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    bool relu;
  };

  // 3x3 convolution
  class Conv : public Op, protected ConvDesc
  {
  public:
    Conv(const ConvDesc& desc);
    
    TensorDesc getDstDesc() const { return dstDesc; }
    virtual void setSrc(const std::shared_ptr<Tensor>& src);
    virtual void setDst(const std::shared_ptr<Tensor>& dst);
    std::shared_ptr<Tensor> getDst() const { return dst; }

  protected:
    TensorDesc dstDesc;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

} // namespace oidn
