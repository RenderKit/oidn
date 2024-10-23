// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUConv final : public Conv
  {
  public:
    CPUConv(CPUEngine* engine, const ConvDesc& desc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    CPUEngine* engine;
    int blockOCB; // block of output channel blocks
    int blockOW;  // block of output width
    int OCBB;     // number of output channel block blocks
    int OWT;      // number of output width tiles
  };

OIDN_NAMESPACE_END