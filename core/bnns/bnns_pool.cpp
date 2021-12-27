// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_pool.h"

namespace oidn {

  BNNSPoolNode::BNNSPoolNode(const Ref<BNNSDevice>& device, const PoolDesc& desc)
    : BNNSNode(device, desc.name),
      PoolNode(desc)
  {
    BNNSLayerParametersPooling params = {
      .i_desc = toNDArrayDesc(*src),
      .o_desc = toNDArrayDesc(*dst),
      .pooling_function = BNNSPoolingFunctionMax,
      .k_width  = 2,
      .k_height = 2,
      .x_stride = 2,
      .y_stride = 2
    };

    filter = BNNSFilterCreateLayerPooling(&params, nullptr);
    if (!filter)
      throw Exception(Error::Unknown, "BNNSFilterCreateLayerPooling failed");
  }

  void BNNSPoolNode::execute()
  {
    BNNSFilterApply(filter, src->data(), dst->data());
  }

} // namespace oidn