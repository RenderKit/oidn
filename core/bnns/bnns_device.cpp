// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_device.h"
#include "bnns_conv.h"
#include "bnns_pool.h"

namespace oidn {

  std::shared_ptr<Conv> BNNSDevice::newConv(const ConvDesc& desc)
  {
    return std::make_shared<BNNSConv>(Ref<BNNSDevice>(this), desc);
  }

  std::shared_ptr<Pool> BNNSDevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<BNNSPool>(Ref<BNNSDevice>(this), desc);
  }

} // namespace oidn
