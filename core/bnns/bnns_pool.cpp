// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_pool.h"

namespace oidn {

  BNNSPool::BNNSPool(const Ref<BNNSDevice>& device, const PoolDesc& desc)
    : Pool(desc),
      device(device) {}

  BNNSPool::~BNNSPool()
  {
    if (filter)
      BNNSFilterDestroy(filter);
  }

  void BNNSPool::finalize()
  {
    BNNSLayerParametersPooling params = {
      .i_desc = toBNNS(srcDesc),
      .o_desc = toBNNS(dstDesc),
      .pooling_function = BNNSPoolingFunctionMax,
      .k_width  = 2,
      .k_height = 2,
      .x_stride = 2,
      .y_stride = 2
    };

    filter = BNNSFilterCreateLayerPooling(&params, nullptr);
    if (!filter)
      throw std::runtime_error("BNNSFilterCreateLayerPooling failed");
  }

  void BNNSPool::run()
  {
    BNNSFilterApply(filter, src->getData(), dst->getData());
  }

} // namespace oidn