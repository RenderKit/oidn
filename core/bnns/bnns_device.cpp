// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_device.h"
#include "bnns_conv.h"
#include "bnns_pool.h"

namespace oidn {

  std::shared_ptr<ConvNode> BNNSDevice::newConvNode(const ConvDesc& desc)
  {
    return std::make_shared<BNNSConvNode>(Ref<BNNSDevice>(this), desc);
  }

  std::shared_ptr<PoolNode> BNNSDevice::newPoolNode(const PoolDesc& desc)
  {
    return std::make_shared<BNNSPoolNode>(Ref<BNNSDevice>(this), desc);
  }

} // namespace oidn
