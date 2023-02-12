// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/cpu/cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class BNNSEngine final : public CPUEngine
  {
  public:
    explicit BNNSEngine(const Ref<CPUDevice>& device);

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
  };

OIDN_NAMESPACE_END
