// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/autoexposure.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUAutoexposure final : public Autoexposure
  {
  public:
    CPUAutoexposure(CPUEngine* engine, const ImageDesc& srcDesc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    CPUEngine* engine;
  };

OIDN_NAMESPACE_END
