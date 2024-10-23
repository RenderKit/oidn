// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/image_copy.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUImageCopy final : public ImageCopy
  {
  public:
    explicit CPUImageCopy(CPUEngine* engine);

    Engine* getEngine() const override { return engine; }
    void submitKernels(const Ref<CancellationToken>& ct) override;

  private:
    CPUEngine* engine;
  };

OIDN_NAMESPACE_END
