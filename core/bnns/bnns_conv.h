// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "bnns_op.h"

namespace oidn {

  class BNNSConv : public BNNSOp, public Conv
  {
  public:
    BNNSConv(const Ref<BNNSDevice>& device, const ConvDesc& desc);

    void finalize() override;
    void run() override;
  };

} // namespace oidn
