// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_conv.h"

namespace oidn {

  BNNSConv::BNNSConv(const Ref<BNNSDevice>& device, const ConvDesc& desc)
    : BNNSOp(device),
      Conv(desc) {}

  void BNNSConv::finalize()
  {
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

    if (relu)
      params.activation.function = BNNSActivationFunctionRectifiedLinear;
    else
      params.activation.function = BNNSActivationFunctionIdentity;

    filter = BNNSFilterCreateLayerConvolution(&params, nullptr);
    if (!filter)
      throw std::runtime_error("BNNSFilterCreateLayerConvolution failed");
  }

  void BNNSConv::run()
  {
    BNNSFilterApply(filter, src->getData(), dst->getData());
  }

} // namespace oidn