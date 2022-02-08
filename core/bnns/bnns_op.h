// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../op.h"
#include "bnns_device.h"

namespace oidn {

  // BNNS operation base class
  class BNNSOp : public BaseOp<BNNSDevice>
  {
  protected:
    BNNSFilter filter = nullptr;

  public:
    BNNSOp(const Ref<BNNSDevice>& device);
    ~BNNSOp();
  };

} // namespace oidn
