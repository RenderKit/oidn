// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../input_process.h"
#include "cpu_device.h"

namespace oidn {

  class CPUInputProcess final : public InputProcess
  {
  public:
    CPUInputProcess(const Ref<CPUDevice>& device, const InputProcessDesc& desc);
    void run() override;

  private:
    Ref<CPUDevice> device;
  };

} // namespace oidn
