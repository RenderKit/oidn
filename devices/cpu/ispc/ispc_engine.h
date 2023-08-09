// Copyright 2023 Linaro Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class ISPCEngine final : public CPUEngine
  {
  public:
    explicit ISPCEngine(const Ref<CPUDevice>& device);

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
  };

OIDN_NAMESPACE_END
