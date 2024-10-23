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

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    CPUEngine* engine;
  };

OIDN_NAMESPACE_END
