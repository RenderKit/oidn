// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../reorder.h"
#include "dnnl_conv.h"
#include "dnnl_tensor.h"
#include "dnnl_reorder.h"

namespace oidn {

  DNNLConv::DNNLConv(const Ref<DNNLDevice>& device, const ConvDesc& desc)
    : DNNLOp(device),
      Conv(desc)
  {
    const dnnl::memory::dims strides = {1, 1};
    const dnnl::memory::dims padding = {1, 1};

    const dnnl::memory& srcMem = getDNNL(*src);
    const dnnl::memory& dstMem = getDNNL(*dst);

    // Let the convolution primitive choose the weight format
    auto weightDesc = dnnl::memory::desc({ weight->getDims() },
                                          srcMem.get_desc().data_type(),
                                          dnnl::memory::format_tag::any);

    // Let the convolution primitive choose the bias format
    auto biasDesc = dnnl::memory::desc({ bias->getDims() },
                                        srcMem.get_desc().data_type(),
                                        dnnl::memory::format_tag::any);

    auto convDesc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      srcMem.get_desc(),
      weightDesc,
      biasDesc,
      dstMem.get_desc(),
      strides, padding, padding);

    // Incorporate relu
    dnnl::primitive_attr convAttr;
    if (desc.relu)
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

    // Reorder the weight tensor to the final format, if necessary
    if (convPrimDesc.weights_desc() != getDNNL(*weight).get_desc())
    {
      weight = std::make_shared<DNNLTensor>(device, convPrimDesc.weights_desc());
      DNNLReorder(device, {desc.weight, weight}).run();
      device->wait();
    }

    // Reorder the bias tensor to the final format, if necessary
    if (convPrimDesc.bias_desc() != getDNNL(*bias).get_desc())
    {
      bias = std::make_shared<DNNLTensor>(device, convPrimDesc.bias_desc());
      DNNLReorder(device, {desc.bias, bias}).run();
      device->wait();
    }

    prim = dnnl::convolution_forward(convPrimDesc);
    args = {{DNNL_ARG_SRC,     srcMem},
            {DNNL_ARG_WEIGHTS, getDNNL(*weight)},
            {DNNL_ARG_BIAS,    getDNNL(*bias)},
            {DNNL_ARG_DST,     dstMem}};
  }

} // namespace oidn