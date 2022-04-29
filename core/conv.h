// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  // Activation function
  enum class Activation
  {
    None, // identity
    ReLU
  };

  // 3x3 convolution descriptor
  struct ConvDesc
  {
    TensorDesc srcDesc;
    TensorDesc weightDesc;
    TensorDesc biasDesc;
    Activation activation;
  };

  // 3x3 convolution
  class Conv : public Op, protected ConvDesc
  {
  public:
    Conv(const ConvDesc& desc);
    
    TensorDesc getDstDesc() const { return dstDesc; }
    std::shared_ptr<Tensor> getDst() const { return dst; }

    void setSrc(const std::shared_ptr<Tensor>& src);
    void setWeight(const std::shared_ptr<Tensor>& weight);
    void setBias(const std::shared_ptr<Tensor>& bias);
    void setDst(const std::shared_ptr<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateWeight() {}
    virtual void updateBias() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;
  };

} // namespace oidn
