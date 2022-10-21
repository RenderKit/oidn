// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "sycl_engine.h"

namespace oidn {

  class SYCLPool : public Pool
  {
  public:
    SYCLPool(const Ref<SYCLEngine>& engine, const PoolDesc& desc);
    void submit() override;

  private:
    Ref<SYCLEngine> engine;
  };

} // namespace oidn
