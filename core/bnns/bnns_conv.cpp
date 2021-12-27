// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_conv.h"

namespace oidn {

  BNNSConvNode::BNNSConvNode(const Ref<BNNSDevice>& device, const ConvDesc& desc)
    : BNNSNode(device, desc.name),
      ConvNode(desc)
  {
    BNNSLayerParametersConvolution params = {
      .i_desc = toNDArrayDesc(*src),
      .w_desc = toNDArrayDesc(*weights),
      .o_desc = toNDArrayDesc(*dst),
      .bias   = toNDArrayDesc(*bias),
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

  void BNNSConvNode::execute()
  {
    BNNSFilterApply(filter, src->data(), dst->data());
  }

} // namespace oidn