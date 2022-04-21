// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "bnns_common.h"

namespace oidn {

  class BNNSPool : public Pool
  {
  public:
    BNNSPool(const Ref<BNNSDevice>& device, const PoolDesc& desc);
    ~BNNSPool();

    void finalize() override;
    void run() override;

  private:
    Ref<BNNSDevice> device;
    BNNSFilter filter = nullptr;
  };

} // namespace oidn
