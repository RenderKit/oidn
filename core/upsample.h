// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  struct UpsampleDesc
  {
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

  // 2x2 nearest-neighbor upsampling
  class Upsample : public virtual Op
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    Upsample(const UpsampleDesc& desc)
      : src(desc.src),
        dst(desc.dst)
    {
      assert(src->getRank() == 3);
      assert(dst->getRank() == 3);
      assert(dst->getLayout() == src->getLayout());
      assert(dst->getC() == src->getC());
      assert(dst->getH() == src->getH() * 2);
      assert(dst->getW() == src->getW() * 2);
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
