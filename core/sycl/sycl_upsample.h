// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "sycl_device.h"

namespace oidn {

  class SYCLUpsample : public Upsample
  {
  public:
    SYCLUpsample(const Ref<SYCLDevice>& device, const UpsampleDesc& desc);
    void run() override;

  private:
    Ref<SYCLDevice> device;
  };

} // namespace oidn
