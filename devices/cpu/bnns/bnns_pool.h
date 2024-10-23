// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/pool.h"
#include "bnns_common.h"

OIDN_NAMESPACE_BEGIN

  class BNNSPool : public Pool
  {
  public:
    BNNSPool(BNNSEngine* engine, const PoolDesc& desc);
    ~BNNSPool();

    Engine* getEngine() const override { return engine; }
    void finalize() override;
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    BNNSEngine* engine;
    BNNSFilter filter = nullptr;
  };

OIDN_NAMESPACE_END
