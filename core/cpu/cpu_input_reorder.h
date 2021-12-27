// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../input_reorder.h"
#include "cpu_node.h"

namespace oidn {

  class CPUInputReorderNode : public CPUNode, public InputReorderNode
  {
  public:
    CPUInputReorderNode(const Ref<CPUDevice>& device, const InputReorderDesc& desc);

    void execute() override;
  };

} // namespace oidn
