// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Accelerate/Accelerate.h>
#include "../device.h"

namespace oidn {

  class BNNSDevice : public Device
  {
  public:
    // Nodes
    std::shared_ptr<ConvNode> newConvNode(const ConvDesc& desc) override;
    std::shared_ptr<PoolNode> newPoolNode(const PoolDesc& desc) override;
  };

} // namespace oidn
