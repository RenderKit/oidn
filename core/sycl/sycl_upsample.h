// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "sycl_engine.h"

namespace oidn {

  class SYCLUpsample : public Upsample
  {
  public:
    SYCLUpsample(const Ref<SYCLEngine>& engine, const UpsampleDesc& desc);
    void submit() override;

  private:
    Ref<SYCLEngine> engine;
  };

} // namespace oidn
