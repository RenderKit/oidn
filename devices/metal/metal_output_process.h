// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/output_process.h"
#include "metal_common.h"
#include "metal_kernel_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalOutputProcess final : public OutputProcess
  {
  public:
    MetalOutputProcess(const Ref<MetalEngine>& engine, const OutputProcessDesc& desc);
    ~MetalOutputProcess();

    void finalize() override;
    void submit() override;

  private:
    ProcessParams createProcessParams();
    void cleanup();

  private:
    Ref<MetalEngine> engine;

    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> paramsBuffer;
  };

OIDN_NAMESPACE_END
