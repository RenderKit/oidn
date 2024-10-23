// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_conv.h"

OIDN_NAMESPACE_BEGIN

  BNNSConv::BNNSConv(BNNSEngine* engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {}

  BNNSConv::~BNNSConv()
  {
    if (filter)
      BNNSFilterDestroy(filter);
  }

  void BNNSConv::updateWeight()
  {
    if (filter)
      throw std::logic_error("convolution weight cannot be set after finalization");
  }

  void BNNSConv::updateBias()
  {
    if (filter)
      throw std::logic_error("convolution bias cannot be set after finalization");
  }

  void BNNSConv::finalize()
  {
    if (filter)
      throw std::logic_error("convolution already finalized");
    if (!weight || !bias)
      throw std::logic_error("convolution weight/bias not set before finalization");

    BNNSLayerParametersConvolution params = {
      .i_desc = toBNNS(srcDesc),
      .w_desc = toBNNS(weight),
      .o_desc = toBNNS(dstDesc),
      .bias   = toBNNS(bias),
      .x_stride = 1,
      .y_stride = 1,
      .x_dilation_stride = 1,
      .y_dilation_stride = 1,
      .x_padding = 1,
      .y_padding = 1,
    };

    if (activation == Activation::ReLU)
      params.activation.function = BNNSActivationFunctionRectifiedLinear;
    else
      params.activation.function = BNNSActivationFunctionIdentity;

    filter = BNNSFilterCreateLayerConvolution(&params, nullptr);
    if (!filter)
      throw std::runtime_error("BNNSFilterCreateLayerConvolution failed");
  }

  void BNNSConv::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!filter)
      throw std::logic_error("convolution not finalized");
    if (!src || !dst)
      throw std::logic_error("convolution source/destination not set");

    void* srcPtr = src->getPtr();
    void* dstPtr = dst->getPtr();
    engine->submitFunc([=] { BNNSFilterApply(filter, srcPtr, dstPtr); }, ct);
  }

OIDN_NAMESPACE_END