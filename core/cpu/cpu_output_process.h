// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../output_process.h"
#include "cpu_engine.h"

namespace oidn {

  class CPUOutputProcess final : public OutputProcess
  {
  public:
    CPUOutputProcess(const Ref<CPUEngine>& engine, const OutputProcessDesc& desc);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

} // namespace oidn
