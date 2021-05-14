// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

#if defined(OIDN_DNNL)

  // Reorder node
  class ReorderNode : public DNNLNode
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    ReorderNode(const Ref<Device>& device,
                const std::shared_ptr<Tensor>& src,
                const std::shared_ptr<Tensor>& dst)
      : DNNLNode(device),
        src(src), dst(dst)
    {
      prim = dnnl::reorder(dnnl::reorder::primitive_desc(src->mem, dst->mem));
      args = {{DNNL_ARG_SRC, src->mem},
              {DNNL_ARG_DST, dst->mem}};
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

#endif

} // namespace oidn
