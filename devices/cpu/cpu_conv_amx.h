// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUConvAMX final : public Conv
  {
  public:
    CPUConvAMX(CPUEngine* engine, const ConvDesc& desc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    CPUEngine* engine;
};

OIDN_NAMESPACE_END