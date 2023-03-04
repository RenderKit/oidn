// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_conv.h"
#include "dnnl_tensor.h"

OIDN_NAMESPACE_BEGIN

  DNNLConv::DNNLConv(const Ref<DNNLEngine>& engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {
    const dnnl::memory::dims strides = {1, 1};
    const dnnl::memory::dims padding = {1, 1};

    // Incorporate activation
    dnnl::primitive_attr attr;
    if (activation == Activation::ReLU)
    {
      dnnl::post_ops ops;
      ops.append_eltwise(
        dnnl::algorithm::eltwise_relu,
        0.f, // alpha
        0.f  // beta
      );
      attr.set_post_ops(ops);
    }
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    primDesc = dnnl::convolution_forward::primitive_desc(
      engine->getDNNLEngine(),
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      toDNNL(srcDesc),
      toDNNL(weightDesc),
      toDNNL(biasDesc),
      toDNNL(dstDesc),
      strides, padding, padding,
      attr);
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
    args[DNNL_ARG_WEIGHTS] = getDNNL(weight);
  }

  void DNNLConv::updateBias()
  {
    args[DNNL_ARG_BIAS] = getDNNL(bias);
  }

  void DNNLConv::updateDst()
  {
    args[DNNL_ARG_DST] = getDNNL(dst);
  }

  void DNNLConv::finalize()
  {
    prim = dnnl::convolution_forward(primDesc);
  }

  void DNNLConv::submit()
  {
    if (!prim)
      throw std::logic_error("convolution not finalized");
    if (!src || !dst || !weight || !bias)
      throw std::logic_error("convolution source/weight/bias/destination not set");

    prim.execute(engine->getDNNLStream(), args);
  }

OIDN_NAMESPACE_END