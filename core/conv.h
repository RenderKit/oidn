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
  class Conv : public BaseOp, protected ConvDesc
  {
  public:
    Conv(const ConvDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    Ref<Tensor> getDst() const { return dst; }

    void setSrc(const Ref<Tensor>& src);
    void setWeight(const Ref<Tensor>& weight);
    void setBias(const Ref<Tensor>& bias);
    void setDst(const Ref<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateWeight() {}
    virtual void updateBias() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;
    Ref<Tensor> src;
    Ref<Tensor> weight;
    Ref<Tensor> bias;
    Ref<Tensor> dst;
  };

OIDN_NAMESPACE_END
