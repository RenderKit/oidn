// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "upsample.h"

OIDN_NAMESPACE_BEGIN

  Upsample::Upsample(const UpsampleDesc& desc)
    : UpsampleDesc(desc)
  {
    if (srcDesc.getRank() != 3)
      throw std::invalid_argument("invalid upsampling source shape");

    TensorDims dstDims{srcDesc.getC(), srcDesc.getH() * 2, srcDesc.getW() * 2};
    TensorDims dstPaddedDims{srcDesc.getPaddedC(), dstDims[1], dstDims[2]};
    dstDesc = {dstDims, dstPaddedDims, srcDesc.layout, srcDesc.dataType};
  }

  void Upsample::setSrc(const Ref<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid upsampling source");

    this->src = src;
    updateSrc();
  }

  void Upsample::setDst(const Ref<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid upsampling destination");

    this->dst = dst;
    updateDst();
  }

OIDN_NAMESPACE_END
