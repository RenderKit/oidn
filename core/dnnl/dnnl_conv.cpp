// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_conv.h"
#include "../reorder.h"
#include "dnnl_tensor.h"
#include "dnnl_reorder.h"

namespace oidn {

  DNNLConv::DNNLConv(const Ref<DNNLDevice>& device, const ConvDesc& desc)
    : Conv(desc),
      device(device)
  {
    const dnnl::memory::dims strides = {1, 1};
    const dnnl::memory::dims padding = {1, 1};

    // Let the convolution primitive choose the weight format
    auto anyWeightDesc = dnnl::memory::desc({ weightDesc.dims },
                                             toDNNL(srcDesc.dataType),
                                             dnnl::memory::format_tag::any
                                             /*dnnl::memory::format_tag::OIhw16i16o*/);

    // Let the convolution primitive choose the bias format
    auto anyBiasDesc = dnnl::memory::desc({ biasDesc.dims },
                                           toDNNL(srcDesc.dataType),
                                           dnnl::memory::format_tag::any);

    auto convDesc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      toDNNL(srcDesc),
      anyWeightDesc,
      anyBiasDesc,
      toDNNL(dstDesc),
      strides, padding, padding);

    // Incorporate activation
    dnnl::primitive_attr convAttr;
    if (activation == Activation::ReLU)
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

    primDesc = dnnl::convolution_forward::primitive_desc(convDesc, convAttr, device->getDNNLEngine());
  }

  size_t DNNLConv::getScratchByteSize() const
  {
    return primDesc.scratchpad_desc().get_size();
  }

  void DNNLConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
    args[DNNL_ARG_SCRATCHPAD] = getDNNL(scratch);
  }

  void DNNLConv::updateSrc()
  {
    args[DNNL_ARG_SRC] = getDNNL(src);
  }

  void DNNLConv::updateWeight()
  {
    if (prim)
      throw std::logic_error("convolution weight cannot be set after finalization");
  }

  void DNNLConv::updateBias()
  {
    if (prim)
      throw std::logic_error("convolution bias cannot be set after finalization");
  }

  void DNNLConv::updateDst()
  {
    args[DNNL_ARG_DST] = getDNNL(dst);
  }

  void DNNLConv::finalize()
  {
    if (prim)
      throw std::logic_error("convolution already finalized");
    if (!weight || !bias)
      throw std::logic_error("convolution weight/bias not set before finalization");

    // Reorder the weight tensor to the final format, if necessary
    if (getDNNL(weight).get_desc() != primDesc.weights_desc())
    {
      auto newWeight = std::make_shared<DNNLTensor>(device, primDesc.weights_desc());
      DNNLReorder(device, {weight, newWeight}).run();
      device->wait();
      weight = newWeight;
    }

    // Reorder the bias tensor to the final format, if necessary
    if (getDNNL(bias).get_desc() != primDesc.bias_desc())
    {
      auto newBias = std::make_shared<DNNLTensor>(device, primDesc.bias_desc());
      DNNLReorder(device, {bias, newBias}).run();
      device->wait();
      bias = newBias;
    }

    args[DNNL_ARG_WEIGHTS] = getDNNL(weight);
    args[DNNL_ARG_BIAS] = getDNNL(bias);

    prim = dnnl::convolution_forward(primDesc);
  }

  void DNNLConv::run()
  {
    if (!prim)
      throw std::logic_error("convolution not finalized");
    if (!src || !dst)
      throw std::logic_error("convolution source/destination not set");

    prim.execute(device->getDNNLStream(), args);
  }

} // namespace oidn