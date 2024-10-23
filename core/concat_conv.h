// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv.h"

OIDN_NAMESPACE_BEGIN

  // Concatenation + convolution descriptor
  struct ConcatConvDesc
  {
    TensorDesc src1Desc;
    TensorDesc src2Desc;
    TensorDesc weightDesc;
    TensorDesc biasDesc;
    Activation activation;
    bool fastMath; // prefer performance over accuracy
  };

  class ConcatConv : public BaseOp, protected ConcatConvDesc
  {
  public:
    ConcatConv(const ConcatConvDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    Ref<Tensor> getDst() const { return dst; }

    void setSrc(const Ref<Tensor>& src1, const Ref<Tensor>& src2);
    void setBias(const Ref<Tensor>& bias);
    void setDst(const Ref<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateBias() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;

    Ref<Tensor> src1;
    Ref<Tensor> src2;
    Ref<Tensor> bias;
    Ref<Tensor> dst;
  };

OIDN_NAMESPACE_END
