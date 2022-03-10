// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "upsample.h"

namespace oidn {

  Upsample::Upsample(const UpsampleDesc& desc)
    : UpsampleDesc(desc)
  {
    assert(srcDesc.getRank() == 3);

    TensorDims dstDims {srcDesc.getC(), srcDesc.getH() * 2, srcDesc.getW() * 2};
    dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  void Upsample::setSrc(const std::shared_ptr<Tensor>& src)
  {
    assert(src->getDesc() == srcDesc);
    this->src = src;
  }

  void Upsample::setDst(const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == dstDesc);
    this->dst = dst;
  }

} // namespace oidn
