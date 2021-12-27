// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../reorder.h"
#include "dnnl_conv.h"
#include "dnnl_tensor.h"
#include "dnnl_reorder.h"

namespace oidn {

  DNNLConvNode::DNNLConvNode(const Ref<DNNLDevice>& device, const ConvDesc& desc)
    : DNNLNode(device, desc.name),
      ConvNode(desc)
  {
    const dnnl::memory::dims strides = {1, 1};
    const dnnl::memory::dims padding = {1, 1};

    const dnnl::memory& srcMem = DNNLTensor::getMemory(*src);
    const dnnl::memory& dstMem = DNNLTensor::getMemory(*dst);

    // Let the convolution primitive choose the weights format
    auto weightsDesc = dnnl::memory::desc({ weights->dims },
                                          srcMem.get_desc().data_type(),
                                          dnnl::memory::format_tag::any);

    // Let the convolution primitive choose the bias format
    auto biasDesc = dnnl::memory::desc({ bias->dims },
                                        srcMem.get_desc().data_type(),
                                        dnnl::memory::format_tag::any);

    auto convDesc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      srcMem.get_desc(),
      weightsDesc,
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

    // Reorder the weights to the final format, if necessary
    if (convPrimDesc.weights_desc() != DNNLTensor::getMemory(*weights).get_desc())
    {
      weights = std::make_shared<DNNLTensor>(device, convPrimDesc.weights_desc());
      DNNLReorderNode(device, {"weightsReorder", desc.weights, weights}).execute();
      device->wait();
    }

    // Reorder the bias to the final format, if necessary
    if (convPrimDesc.bias_desc() != DNNLTensor::getMemory(*bias).get_desc())
    {
      bias = std::make_shared<DNNLTensor>(device, convPrimDesc.bias_desc());
      DNNLReorderNode(device, {"biasReorder", desc.bias, bias}).execute();
      device->wait();
    }

    prim = dnnl::convolution_forward(convPrimDesc);
    args = {{DNNL_ARG_SRC,     srcMem},
            {DNNL_ARG_WEIGHTS, DNNLTensor::getMemory(*weights)},
            {DNNL_ARG_BIAS,    DNNLTensor::getMemory(*bias)},
            {DNNL_ARG_DST,     dstMem}};
  }

} // namespace oidn