// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class BNNSEngine final : public CPUEngine
  {
  public:
    BNNSEngine(CPUDevice* device, int numThreads);

    // Ops
    Ref<Conv> newConv(const ConvDesc& desc) override;
    Ref<Pool> newPool(const PoolDesc& desc) override;
  };

OIDN_NAMESPACE_END
