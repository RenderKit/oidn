// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(OIDN_COMPILE_METAL_DEVICE)
  #include "op.h"
  #include "image.h"
  #include "record.h"
#endif

OIDN_NAMESPACE_BEGIN

  struct AutoexposureParams
  {
    static constexpr oidn_constant int maxBinSize = 16;
    static constexpr oidn_constant float key = 0.18f;
    static constexpr oidn_constant float eps = 1e-8f;
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE)

  class Autoexposure : public BaseOp, public AutoexposureParams
  {
  public:
    explicit Autoexposure(const ImageDesc& srcDesc)
      : srcDesc(srcDesc)
    {
      numBinsH = ceil_div(srcDesc.getH(), maxBinSize);
      numBinsW = ceil_div(srcDesc.getW(), maxBinSize);
      numBins = numBinsH * numBinsW;
    }

    void setSrc(const Ref<Image>& src)
    {
      if (!src || src->getW() != srcDesc.getW() || src->getH() != srcDesc.getH())
        throw std::invalid_argument("invalid autoexposure source");
      this->src = src;
    }

    void setDst(const Ref<Record<float>>& dst) { this->dst = dst; }
    float* getDstPtr() const { return dst->getPtr(); }

  protected:
    ImageDesc srcDesc;
    Ref<Image> src;
    Ref<Record<float>> dst;

    int numBinsH;
    int numBinsW;
    int numBins;
  };

#endif // !defined(OIDN_COMPILE_METAL_DEVICE)

OIDN_NAMESPACE_END
