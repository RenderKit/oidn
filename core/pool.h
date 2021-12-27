// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  // 2x2 max pooling descriptor
  struct PoolDesc
  {
    std::string name;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

  // 2x2 max pooling node
  class PoolNode : public virtual Node
  {
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    PoolNode(const PoolDesc& desc)
      : src(desc.src),
        dst(desc.dst)
    {
      assert(src->ndims() == 3);
      assert(dst->ndims() == 3);
      assert(dst->layout == src->layout);
      assert(src->dims[0] == dst->dims[0]);     // C
      assert(src->dims[1] == dst->dims[1] * 2); // H
      assert(src->dims[2] == dst->dims[2] * 2); // W
    }

    std::shared_ptr<Tensor> getDst() const { return dst; }
  };

} // namespace oidn
