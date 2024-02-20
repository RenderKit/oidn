// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_engine.h"
#include "bnns_conv.h"
#include "bnns_pool.h"

OIDN_NAMESPACE_BEGIN

  BNNSEngine::BNNSEngine(CPUDevice* device, int numThreads)
    : CPUEngine(device, numThreads)
  {}

  Ref<Conv> BNNSEngine::newConv(const ConvDesc& desc)
  {
    return makeRef<BNNSConv>(this, desc);
  }

  Ref<Pool> BNNSEngine::newPool(const PoolDesc& desc)
  {
    return makeRef<BNNSPool>(this, desc);
  }

OIDN_NAMESPACE_END
