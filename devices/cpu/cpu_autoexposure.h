// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/autoexposure.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUAutoexposure final : public Autoexposure
  {
  public:
    CPUAutoexposure(const Ref<CPUEngine>& engine, const ImageDesc& srcDesc);
    void submit() override;
    const float* getResult() const override { return &result; }

  private:
    Ref<CPUEngine> engine;
    float result;
  };

OIDN_NAMESPACE_END
