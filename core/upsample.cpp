// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "upsample.h"

namespace oidn {

  Upsample::Upsample(const UpsampleDesc& desc)
    : UpsampleDesc(desc)
  {
    if (srcDesc.getRank() != 3)
      throw std::invalid_argument("invalid upsampling source shape");

    TensorDims dstDims {srcDesc.getC(), srcDesc.getH() * 2, srcDesc.getW() * 2};
    dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  void Upsample::setSrc(const std::shared_ptr<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid upsampling source");

    this->src = src;
    updateSrc();
  }

  void Upsample::setDst(const std::shared_ptr<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid upsampling destination");

    this->dst = dst;
    updateDst();
  }

} // namespace oidn
