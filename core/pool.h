// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  // 2x2 max pooling descriptor
  struct PoolDesc
  {
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

  // 2x2 max pooling
  class Pool : public virtual Op
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    Pool(const PoolDesc& desc)
      : src(desc.src),
        dst(desc.dst)
    {
      assert(src->getRank() == 3);
      assert(dst->getRank() == 3);
      assert(dst->getLayout() == src->getLayout());
      assert(src->getC() == dst->getC());
      assert(src->getH() == dst->getH() * 2);
      assert(src->getW() == dst->getW() * 2);
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
