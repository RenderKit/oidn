// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "bnns_node.h"

namespace oidn {

  class BNNSConvNode : public BNNSNode, public ConvNode
  {
  public:
    BNNSConvNode(const Ref<BNNSDevice>& device, const ConvDesc& desc);

    void execute() override;
  };

} // namespace oidn
