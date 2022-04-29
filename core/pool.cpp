// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pool.h"

namespace oidn {

  Pool::Pool(const PoolDesc& desc)
    : PoolDesc(desc)
  {
    if (srcDesc.getRank() != 3 || srcDesc.getH() % 2 != 0 || srcDesc.getW() % 2 != 0)
      throw std::invalid_argument("invalid pooling source shape");
  
    TensorDims dstDims {srcDesc.getC(), srcDesc.getH() / 2, srcDesc.getW() / 2};
    dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  void Pool::setSrc(const std::shared_ptr<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid pooling source");

    this->src = src;
    updateSrc();
  }

  void Pool::setDst(const std::shared_ptr<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid pooling destination");

    this->dst = dst;
    updateDst();
  }

} // namespace oidn
