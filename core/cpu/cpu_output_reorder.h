// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../output_reorder.h"
#include "cpu_node.h"

namespace oidn {

  class CPUOutputReorderNode : public CPUNode, public OutputReorderNode
  {
  public:
    CPUOutputReorderNode(const Ref<CPUDevice>& device, const OutputReorderDesc& desc);

    void execute() override;
  };

} // namespace oidn
