// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "bnns_common.h"

namespace oidn {

  class BNNSConv : public Conv
  {
  public:
    BNNSConv(const Ref<BNNSEngine>& engine, const ConvDesc& desc);
    ~BNNSConv();

    void finalize() override;
    void submit() override;

  private:
    void updateWeight() override;
    void updateBias() override;

    Ref<BNNSEngine> engine;
    BNNSFilter filter = nullptr;
  };

} // namespace oidn
