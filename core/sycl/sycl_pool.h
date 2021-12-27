// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "sycl_node.h"

namespace oidn {

  class SYCLPoolNode : public SYCLNode, public PoolNode
  {
  public:
    SYCLPoolNode(const Ref<SYCLDevice>& device, const PoolDesc& desc);

    void execute() override;
  };

} // namespace oidn
