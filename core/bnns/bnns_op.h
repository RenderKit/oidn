// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../op.h"
#include "bnns_device.h"

namespace oidn {

  // BNNS operation base class
  class BNNSOp : public BaseOp<BNNSDevice>
  {
  public:
    BNNSOp(const Ref<BNNSDevice>& device);
    ~BNNSOp();

  protected:
    BNNSFilter filter = nullptr;
  };

} // namespace oidn
