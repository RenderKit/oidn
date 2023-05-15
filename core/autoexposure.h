// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"

OIDN_NAMESPACE_BEGIN

  class Autoexposure : public Op
  {
  public:
    static constexpr int maxBinSize = 16;
    static constexpr float key = 0.18f;
    static constexpr float eps = 1e-8f;

    Autoexposure(const ImageDesc& srcDesc)
      : srcDesc(srcDesc)
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

    // Returns pointer to the result in device memory
    virtual const float* getResult() const = 0;

  protected:
    ImageDesc srcDesc;
    std::shared_ptr<Image> src;

    int numBinsH;
    int numBinsW;
    int numBins;
  };

OIDN_NAMESPACE_END
