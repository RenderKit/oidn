// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  class Autoexposure : public Op
  {
  public:
    static constexpr int maxBinSize = 16;
    static constexpr float key = 0.18f;
    static constexpr float eps = 1e-8f;

    explicit Autoexposure(const ImageDesc& srcDesc)
      : srcDesc(srcDesc),
        dstDesc({1}, TensorLayout::x, DataType::Float32)
    {
      numBinsH = ceil_div(srcDesc.getH(), maxBinSize);
      numBinsW = ceil_div(srcDesc.getW(), maxBinSize);
      numBins = numBinsH * numBinsW;
    }

    void setSrc(const std::shared_ptr<Image>& src)
    {
      if (!src || src->getW() != srcDesc.getW() || src->getH() != srcDesc.getH())
        throw std::invalid_argument("invalid autoexposure source");
      this->src = src;
    }

    void setDst(const std::shared_ptr<Tensor>& dst)
    {
      if (!dst || dst->getDesc() != dstDesc)
        throw std::invalid_argument("invalid autoexposure destination");
      this->dst = dst;
    }

    TensorDesc getDstDesc() const { return dstDesc; }

    float* getDstPtr() const
    {
      return static_cast<float*>(dst->getPtr());
    }

  protected:
    ImageDesc srcDesc;
    TensorDesc dstDesc;
    std::shared_ptr<Image> src;
    std::shared_ptr<Tensor> dst;

    int numBinsH;
    int numBinsW;
    int numBins;
  };

OIDN_NAMESPACE_END
