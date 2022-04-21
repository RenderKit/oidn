// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../output_process.h"
#include "cpu_device.h"

namespace oidn {

  class CPUOutputProcess final : public OutputProcess
  {
  public:
    CPUOutputProcess(const Ref<CPUDevice>& device, const OutputProcessDesc& desc);
    void run() override;

  private:
    Ref<CPUDevice> device;
  };

} // namespace oidn
