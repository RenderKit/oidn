// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  struct ReorderTile
  {
    int hSrcBegin;
    int wSrcBegin;
    int hDstBegin;
    int wDstBegin;
    int H;
    int W;

    operator ispc::ReorderTile() const
    {
      ispc::ReorderTile res;
      res.hSrcBegin = hSrcBegin;
      res.wSrcBegin = wSrcBegin;
      res.hDstBegin = hDstBegin;
      res.wDstBegin = wDstBegin;
      res.H = H;
      res.W = W;
      return res;
    }
  };

#if defined(OIDN_DNNL)

  // Reorder node
  class ReorderNode : public DNNLNode
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    ReorderNode(const Ref<Device>& device,
                const std::string& name,
                const std::shared_ptr<Tensor>& src,
                const std::shared_ptr<Tensor>& dst)
      : DNNLNode(device, name),
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
