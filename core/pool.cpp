// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pool.h"

OIDN_NAMESPACE_BEGIN

  Pool::Pool(const PoolDesc& desc)
    : PoolDesc(desc)
  {
    if (srcDesc.getRank() != 3 || srcDesc.getH() % 2 != 0 || srcDesc.getW() % 2 != 0)
      throw std::invalid_argument("invalid pooling source shape");

    TensorDims dstDims{srcDesc.getC(), srcDesc.getH() / 2, srcDesc.getW() / 2};
    TensorDims dstPaddedDims{srcDesc.getPaddedC(), dstDims[1], dstDims[2]};
    dstDesc = {dstDims, dstPaddedDims, srcDesc.layout, srcDesc.dataType};
  }

  void Pool::setSrc(const Ref<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid pooling source");

    this->src = src;
    updateSrc();
  }

  void Pool::setDst(const Ref<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid pooling destination");

    this->dst = dst;
    updateDst();
  }

OIDN_NAMESPACE_END
