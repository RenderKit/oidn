// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/upsample.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUUpsample final : public Upsample
  {
  public:
    CPUUpsample(CPUEngine* engine, const UpsampleDesc& desc);
    void submit() override;

  private:
    CPUEngine* engine;
  };

OIDN_NAMESPACE_END
