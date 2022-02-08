// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "sycl_op.h"

namespace oidn {

  class SYCLPool : public SYCLOp, public Pool
  {
  public:
    SYCLPool(const Ref<SYCLDevice>& device, const PoolDesc& desc);

    void run() override;
  };

} // namespace oidn
