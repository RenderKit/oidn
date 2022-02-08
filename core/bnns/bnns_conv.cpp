// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_conv.h"

namespace oidn {

  BNNSConv::BNNSConv(const Ref<BNNSDevice>& device, const ConvDesc& desc)
    : BNNSOp(device),
      Conv(desc)
  {
    BNNSLayerParametersConvolution params = {
      .i_desc = toBNNS(*src),
      .w_desc = toBNNS(*weight),
      .o_desc = toBNNS(*dst),
      .bias   = toBNNS(*bias),
      .x_stride = 1,
      .y_stride = 1,
      .x_dilation_stride = 1,
      .y_dilation_stride = 1,
      .x_padding = 1,
      .y_padding = 1,
    };

    if (desc.relu)
      params.activation.function = BNNSActivationFunctionRectifiedLinear;
    else
      params.activation.function = BNNSActivationFunctionIdentity;

    filter = BNNSFilterCreateLayerConvolution(&params, nullptr);
    if (!filter)
      throw Exception(Error::Unknown, "BNNSFilterCreateLayerConvolution failed");
  }

  void BNNSConv::run()
  {
    BNNSFilterApply(filter, src->getData(), dst->getData());
  }

} // namespace oidn