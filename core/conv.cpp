// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conv.h"

namespace oidn {

  Conv::Conv(const ConvDesc& desc)
    : ConvDesc(desc)
  {
    assert(srcDesc.getRank() == 3);
    assert(weight->getRank() == 4);
    assert(weight->getI() == srcDesc.getC());
    assert(bias->getRank() == 1);
    assert(bias->getX() == weight->getO());

    TensorDims dstDims {weight->getO(), srcDesc.getH(), srcDesc.getW()};
    dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  void Conv::setSrc(const std::shared_ptr<Tensor>& src)
  {
    assert(src->getDesc() == srcDesc);
    this->src = src;
  }

  void Conv::setDst(const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == dstDesc);
    this->dst = dst;
  }

} // namespace oidn
