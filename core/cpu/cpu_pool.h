// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "cpu_engine.h"

namespace oidn {

  class CPUPool final : public Pool
  {
  public:
    CPUPool(const Ref<CPUEngine>& engine, const PoolDesc& desc);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

} // namespace oidn
