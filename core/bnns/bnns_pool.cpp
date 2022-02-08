// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_pool.h"

namespace oidn {

  BNNSPool::BNNSPool(const Ref<BNNSDevice>& device, const PoolDesc& desc)
    : BNNSOp(device),
      Pool(desc)
  {
    BNNSLayerParametersPooling params = {
      .i_desc = toBNNS(*src),
      .o_desc = toBNNS(*dst),
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

  void BNNSPool::run()
  {
    BNNSFilterApply(filter, src->getData(), dst->getData());
  }

} // namespace oidn