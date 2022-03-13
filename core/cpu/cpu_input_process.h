// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../input_process.h"
#include "cpu_op.h"

namespace oidn {

  class CPUInputProcess final : public CPUOp, public InputProcess
  {
  public:
    CPUInputProcess(const Ref<CPUDevice>& device, const InputProcessDesc& desc);

    void run() override;
  };

} // namespace oidn
