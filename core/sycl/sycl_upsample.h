// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "sycl_op.h"

namespace oidn {

  class SYCLUpsample : public SYCLOp, public Upsample
  {
  public:
    SYCLUpsample(const Ref<SYCLDevice>& device, const UpsampleDesc& desc);

    void run() override;
  };

} // namespace oidn
