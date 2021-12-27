// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "cpu_node.h"

namespace oidn {

  class CPUUpsampleNode : public CPUNode, public UpsampleNode
  {
  public:
    CPUUpsampleNode(const Ref<CPUDevice>& device, const UpsampleDesc& desc);

    void execute() override;
  };

} // namespace oidn
