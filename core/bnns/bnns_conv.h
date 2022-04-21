// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "bnns_common.h"

namespace oidn {

  class BNNSConv : public Conv
  {
  public:
    BNNSConv(const Ref<BNNSDevice>& device, const ConvDesc& desc);
    ~BNNSConv();

    void finalize() override;
    void run() override;

  private:
    Ref<BNNSDevice> device;
    BNNSFilter filter = nullptr;
  };

} // namespace oidn
