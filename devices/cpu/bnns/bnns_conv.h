// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "bnns_common.h"

OIDN_NAMESPACE_BEGIN

  class BNNSConv : public Conv
  {
  public:
    BNNSConv(BNNSEngine* engine, const ConvDesc& desc);
    ~BNNSConv();

    Engine* getEngine() const override { return engine; }
    void finalize() override;
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    void updateWeight() override;
    void updateBias() override;

    BNNSEngine* engine;
    BNNSFilter filter = nullptr;
  };

OIDN_NAMESPACE_END
