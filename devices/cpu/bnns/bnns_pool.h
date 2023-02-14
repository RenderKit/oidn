// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/pool.h"
#include "bnns_common.h"

OIDN_NAMESPACE_BEGIN

  class BNNSPool : public Pool
  {
  public:
    BNNSPool(const Ref<BNNSEngine>& engine, const PoolDesc& desc);
    ~BNNSPool();

    void finalize() override;
    void submit() override;

  private:
    Ref<BNNSEngine> engine;
    BNNSFilter filter = nullptr;
  };

OIDN_NAMESPACE_END
