// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pool.h"

namespace oidn {

  Pool::Pool(const PoolDesc& desc)
    : PoolDesc(desc)
  {
    assert(srcDesc.getRank() == 3);
    assert(srcDesc.getH() % 2 == 0);
    assert(srcDesc.getW() % 2 == 0);

    TensorDims dstDims {srcDesc.getC(), srcDesc.getH() / 2, srcDesc.getW() / 2};
    dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  void Pool::setSrc(const std::shared_ptr<Tensor>& src)
  {
    assert(src->getDesc() == srcDesc);
    this->src = src;
  }

  void Pool::setDst(const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == dstDesc);
    this->dst = dst;
  }

} // namespace oidn
