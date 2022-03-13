// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "bnns_op.h"

namespace oidn {

  class BNNSPool : public BNNSOp, public Pool
  {
  public:
    BNNSPool(const Ref<BNNSDevice>& device, const PoolDesc& desc);

    void finalize() override;
    void run() override;
  };

} // namespace oidn
