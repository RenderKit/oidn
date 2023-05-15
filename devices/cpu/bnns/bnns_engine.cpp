// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_engine.h"
#include "bnns_conv.h"
#include "bnns_pool.h"

OIDN_NAMESPACE_BEGIN

  BNNSEngine::BNNSEngine(const Ref<CPUDevice>& device)
    : CPUEngine(device)
  {}

  std::shared_ptr<Conv> BNNSEngine::newConv(const ConvDesc& desc)
  {
    return std::make_shared<BNNSConv>(this, desc);
  }

  std::shared_ptr<Pool> BNNSEngine::newPool(const PoolDesc& desc)
  {
    return std::make_shared<BNNSPool>(this, desc);
  }

OIDN_NAMESPACE_END
