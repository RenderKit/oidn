// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/image_copy.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUImageCopy final : public ImageCopy
  {
  public:
    explicit CPUImageCopy(const Ref<CPUEngine>& engine);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

OIDN_NAMESPACE_END
