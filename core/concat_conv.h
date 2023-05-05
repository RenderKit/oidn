// Copyright 2009-2023 Intel Corporation
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

  class ConcatConv : public Op, protected ConcatConvDesc
  {
  public:
    ConcatConv(const ConcatConvDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    std::shared_ptr<Tensor> getDst() const { return dst; }

    void setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2);
    void setBias(const std::shared_ptr<Tensor>& bias);
    void setDst(const std::shared_ptr<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateBias() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;

    std::shared_ptr<Tensor> src1;
    std::shared_ptr<Tensor> src2;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;
  };

OIDN_NAMESPACE_END
