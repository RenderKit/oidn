// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  // Activation function
  enum class Activation
  {
    None, // identity
    ReLU
  };

  enum class PostOp
  {
    None,
    Pool,
    Upsample
  };

  // Convolution descriptor
  struct ConvDesc
  {
    TensorDesc srcDesc;
    TensorDesc weightDesc;
    TensorDesc biasDesc;
    Activation activation;
    PostOp postOp;
    bool fastMath; // prefer performance over accuracy
  };

  // Convolution
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

OIDN_NAMESPACE_END
