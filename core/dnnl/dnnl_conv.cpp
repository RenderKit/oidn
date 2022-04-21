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
    auto weightDesc = dnnl::memory::desc({ weight->getDims() },
                                          toDNNL(srcDesc.dataType),
                                          dnnl::memory::format_tag::any);

    // Let the convolution primitive choose the bias format
    auto biasDesc = dnnl::memory::desc({ bias->getDims() },
                                        toDNNL(srcDesc.dataType),
                                        dnnl::memory::format_tag::any);

    auto convDesc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      toDNNL(srcDesc),
      weightDesc,
      biasDesc,
      toDNNL(dstDesc),
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

  void DNNLConv::setSrc(const std::shared_ptr<Tensor>& src)
  {
    Conv::setSrc(src);
    args[DNNL_ARG_SRC] = getDNNL(src);
  }

  void DNNLConv::setDst(const std::shared_ptr<Tensor>& dst)
  {
    Conv::setDst(dst);
    args[DNNL_ARG_DST] = getDNNL(dst);
  }

  void DNNLConv::finalize()
  {
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

    prim = dnnl::convolution_forward(primDesc);

    args[DNNL_ARG_WEIGHTS] = getDNNL(weight);
    args[DNNL_ARG_BIAS]    = getDNNL(bias);
  }

  void DNNLConv::run()
  {
    prim.execute(device->getDNNLStream(), args);
  }

} // namespace oidn