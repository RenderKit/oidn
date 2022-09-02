// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "sycl_device.h"

namespace oidn {

  class SYCLConvPVC : public Conv
  {
  public:
    SYCLConvPVC(const Ref<SYCLDevice>& device, const ConvDesc& desc);
    void run() override;

  private:
    template<PostOp kernelPostOp>
    void runImpl();

    Ref<SYCLDevice> device;
  };

} // namespace oidn
