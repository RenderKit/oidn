// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "cpu_op.h"

namespace oidn {

  class CPUUpsample final : public CPUOp, public Upsample
  {
  public:
    CPUUpsample(const Ref<CPUDevice>& device, const UpsampleDesc& desc);

    void run() override;
  };

} // namespace oidn
