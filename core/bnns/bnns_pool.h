// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "bnns_node.h"

namespace oidn {

  class BNNSPoolNode : public BNNSNode, public PoolNode
  {
  public:
    BNNSPoolNode(const Ref<BNNSDevice>& device, const PoolDesc& desc);

    void execute() override;
  };

} // namespace oidn
