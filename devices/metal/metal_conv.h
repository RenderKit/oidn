// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  class MetalConv final : public Conv
  {
  public:
    MetalConv(MetalEngine* engine, const ConvDesc& desc);
    ~MetalConv();

    Engine* getEngine() const override { return engine; }
    void finalize() override;
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    void updateWeight() override;
    void updateBias() override;

    MetalEngine* engine;
    MPSGraph* mpsGraph = nullptr;
    MPSGraphTensor* mpsSrc = nullptr;
    MPSGraphTensor* mpsWeight = nullptr;
    MPSGraphTensor* mpsBias = nullptr;
    MPSGraphTensor* mpsDst = nullptr;
  };

OIDN_NAMESPACE_END
