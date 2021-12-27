// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "sycl_node.h"

namespace oidn {

  class SYCLUpsampleNode : public SYCLNode, public UpsampleNode
  {
  public:
    SYCLUpsampleNode(const Ref<SYCLDevice>& device, const UpsampleDesc& desc);

    void execute() override;
  };

} // namespace oidn
