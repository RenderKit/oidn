// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/input_process.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUInputProcess final : public InputProcess
  {
  public:
    CPUInputProcess(const Ref<CPUEngine>& engine, const InputProcessDesc& desc);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

OIDN_NAMESPACE_END
