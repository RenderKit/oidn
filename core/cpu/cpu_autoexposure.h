// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../autoexposure.h"
#include "cpu_op.h"

namespace oidn {

  class CPUAutoexposure final : public BaseOp<>, public Autoexposure
  {
  public:
    CPUAutoexposure(const Ref<Device>& device, const ImageDesc& srcDesc);
    void run() override;
  };

} // namespace oidn
