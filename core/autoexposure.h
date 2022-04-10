// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"

namespace oidn {

  class Autoexposure : public virtual Op
  {
  public:
    Autoexposure(const ImageDesc& srcDesc)
      : srcDesc(srcDesc),
        result(0)
    {
      numBinsH = ceil_div(srcDesc.getH(), maxBinSize);
      numBinsW = ceil_div(srcDesc.getW(), maxBinSize);
      numBins = numBinsH * numBinsW;
    }
    
    void setSrc(const std::shared_ptr<Image>& src)
    {
      assert(src->getW() == srcDesc.getW() && src->getH() == srcDesc.getH());
      this->src = src;
    }
    
    float getResult() const { return result; }

  protected:
    static constexpr int maxBinSize = 16;

    ImageDesc srcDesc;
    std::shared_ptr<Image> src;

    float result;

    int numBinsH;
    int numBinsW;
    int numBins;
  };

} // namespace oidn
