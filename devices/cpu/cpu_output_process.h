// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/output_process.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUOutputProcess final : public OutputProcess
  {
  public:
    CPUOutputProcess(CPUEngine* engine, const OutputProcessDesc& desc);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    CPUEngine* engine;
  };

OIDN_NAMESPACE_END
