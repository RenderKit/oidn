// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  struct UpsampleDesc
  {
    std::string name;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

  // 2x2 nearest-neighbor upsampling node
  class UpsampleNode : public virtual Node
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    UpsampleNode(const UpsampleDesc& desc)
      : src(desc.src),
        dst(desc.dst)
    {
      assert(src->ndims() == 3);
      assert(dst->ndims() == 3);
      assert(dst->layout == src->layout);
      assert(dst->dims[0] == src->dims[0]);     // C
      assert(dst->dims[1] == src->dims[1] * 2); // H
      assert(dst->dims[2] == src->dims[2] * 2); // W
    }

    std::shared_ptr<Tensor> getDst() const { return dst; }
  };

} // namespace oidn
