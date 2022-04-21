// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "sycl_device.h"

namespace oidn {

  class SYCLPool : public Pool
  {
  public:
    SYCLPool(const Ref<SYCLDevice>& device, const PoolDesc& desc);
    void run() override;

  private:
    Ref<SYCLDevice> device;
  };

} // namespace oidn
