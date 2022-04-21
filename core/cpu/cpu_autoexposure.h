// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../autoexposure.h"
#include "cpu_device.h"

namespace oidn {

  class CPUAutoexposure final : public Autoexposure
  {
  public:
    CPUAutoexposure(const Ref<CPUDevice>& device, const ImageDesc& srcDesc);
    void run() override;
    const float* getResult() const override { return &result; }

  private:
    Ref<CPUDevice> device;
    float result;
  };

} // namespace oidn
