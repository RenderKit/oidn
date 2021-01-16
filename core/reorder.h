// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  // Reorder node
  class ReorderNode : public DNNLNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    ReorderNode(const Ref<Device>& device,
                const Ref<Tensor>& src,
                const Ref<Tensor>& dst)
      : DNNLNode(device),
        src(src), dst(dst)
    {
      setPrimitive(dnnl::reorder(dnnl::reorder::primitive_desc(src->mem, dst->mem)));
      setArgs({{DNNL_ARG_SRC, src->mem},
               {DNNL_ARG_DST, dst->mem}});
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
