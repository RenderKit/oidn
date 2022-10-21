// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "cpu_engine.h"

namespace oidn {

  class CPUUpsample final : public Upsample
  {
  public:
    CPUUpsample(const Ref<CPUEngine>& engine, const UpsampleDesc& desc);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

} // namespace oidn
