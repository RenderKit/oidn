// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

#if defined(OIDN_DNNL)

  // DNNL 2x2 max pooling node
  class PoolNode : public DNNLNode
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    PoolNode(const Ref<Device>& device,
             const std::string& name,
             const std::shared_ptr<Tensor>& src,
             const std::shared_ptr<Tensor>& dst)
      : DNNLNode(device, name),
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

      prim = dnnl::pooling_forward(poolPrimDesc);
      args = {{DNNL_ARG_SRC, src->mem},
              {DNNL_ARG_DST, dst->mem}};
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

#else

  // BNNS 2x2 max pooling node
  class PoolNode : public BNNSNode
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    PoolNode(const Ref<Device>& device,
             const std::string& name,
             const std::shared_ptr<Tensor>& src,
             const std::shared_ptr<Tensor>& dst)
      : BNNSNode(device, name),
        src(src), dst(dst)
    {
      BNNSLayerParametersPooling params = {
        .i_desc = *src,
        .o_desc = *dst,
        .pooling_function = BNNSPoolingFunctionMax,
        .k_width  = 2,
        .k_height = 2,
        .x_stride = 2,
        .y_stride = 2
      };

      filter = BNNSFilterCreateLayerPooling(&params, nullptr);
      if (!filter)
        throw Exception(Error::Unknown, "BNNSFilterCreateLayerPooling failed");
    }

    void execute() override
    {
      BNNSFilterApply(filter, src->data(), dst->data());
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

#endif

} // namespace oidn
