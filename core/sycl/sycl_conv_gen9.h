// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "sycl_device.h"

namespace oidn {

  class SYCLConvGen9 : public Conv
  {
  public:
    SYCLConvGen9(const Ref<SYCLDevice>& device, const ConvDesc& desc);
    void run() override;

  private:
    Ref<SYCLDevice> device;
  };

} // namespace oidn
