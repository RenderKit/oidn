// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "reorder.h"

namespace oidn {

  // Convolution node
  class ConvNode : public DNNLNode
  {
  private:
    Ref<Tensor> src;
    Ref<Tensor> weights;
    Ref<Tensor> bias;
    Ref<Tensor> dst;

  public:
    ConvNode(const Ref<Device>& device,
             const Ref<Tensor>& src,
             const Ref<Tensor>& weights,
             const Ref<Tensor>& bias,
             const Ref<Tensor>& dst,
             bool relu)
      : DNNLNode(device),
        src(src), weights(weights), bias(bias), dst(dst)
    {
      const dnnl::memory::dims strides = {1, 1};
      const dnnl::memory::dims padding = {1, 1};

      // Let the convolution primitive choose the weights format
      auto weightsDesc = dnnl::memory::desc({ weights->dims },
                                            dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::any);

      auto convDesc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        src->mem.get_desc(),
        weightsDesc,
        bias->mem.get_desc(),
        dst->mem.get_desc(),
        strides, padding, padding);

      // Incorporate relu
      dnnl::primitive_attr convAttr;
      if (relu)
      {
        dnnl::post_ops ops;
        ops.append_eltwise(
          1.f,   // scale
          dnnl::algorithm::eltwise_relu,
          0.f,   // alpha
          0.f    // beta
        );
        convAttr.set_post_ops(ops);
      }
      convAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

      auto convPrimDesc = dnnl::convolution_forward::primitive_desc(convDesc, convAttr, device->getDNNLEngine());

      // Reorder the weights to the final format, if necessary
      if (convPrimDesc.weights_desc() != weights->mem.get_desc())
      {
        this->weights = makeRef<Tensor>(device, convPrimDesc.weights_desc());
        ReorderNode(device, weights, this->weights).execute();
      }

      setPrimitive(dnnl::convolution_forward(convPrimDesc));
      setArgs({{DNNL_ARG_SRC,     src->mem},
               {DNNL_ARG_WEIGHTS, this->weights->mem},
               {DNNL_ARG_BIAS,    this->bias->mem},
               {DNNL_ARG_DST,     dst->mem}});
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
