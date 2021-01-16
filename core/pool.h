// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  // Pooling node
  class PoolNode : public DNNLNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    PoolNode(const Ref<Device>& device,
             const Ref<Tensor>& src,
             const Ref<Tensor>& dst)
      : DNNLNode(device),
        src(src), dst(dst)
    {
      const dnnl::memory::dims kernel  = {2, 2};
      const dnnl::memory::dims strides = {2, 2};
      const dnnl::memory::dims padding = {0, 0};

      auto poolDesc = dnnl::pooling_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
        src->mem.get_desc(),
        dst->mem.get_desc(),
        strides, kernel, padding, padding);

      dnnl::primitive_attr poolAttr;
      poolAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      auto poolPrimDesc = dnnl::pooling_forward::primitive_desc(poolDesc, poolAttr, device->getDNNLEngine());

      setPrimitive(dnnl::pooling_forward(poolPrimDesc));
      setArgs({{DNNL_ARG_SRC, src->mem},
               {DNNL_ARG_DST, dst->mem}});
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
