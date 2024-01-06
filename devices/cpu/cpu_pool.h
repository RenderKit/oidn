// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/pool.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUPool final : public Pool
  {
  public:
    CPUPool(CPUEngine* engine, const PoolDesc& desc);
    void submit() override;
  };

OIDN_NAMESPACE_END
