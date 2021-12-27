// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../node.h"
#include "bnns_device.h"

namespace oidn {

  // BNNS node base class
  class BNNSNode : public BaseNode<BNNSDevice>
  {
  protected:
    BNNSFilter filter = nullptr;

    static BNNSNDArrayDescriptor toNDArrayDesc(Tensor& tz);

  public:
    BNNSNode(const Ref<BNNSDevice>& device, const std::string& name);
    ~BNNSNode();
  };

} // namespace oidn
