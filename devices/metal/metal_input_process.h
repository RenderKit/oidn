// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/input_process.h"
#include "metal_common.h"
#include "metal_kernel_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalInputProcess final : public InputProcess
  {
  public:
    MetalInputProcess(const Ref<MetalEngine>& engine, const InputProcessDesc& desc);
    ~MetalInputProcess();

    void finalize() override;
    void submit() override;

  private:
    void cleanup();
    ProcessParams createProcessParams();
    id<MTLBuffer> bufferFromImageOrMain(Image* ptr);

  private:
    Ref<MetalEngine> engine;

    MTLComputePipelineState_t pipeline;
    MTLCommandQueue_t commandQueue;
    id<MTLBuffer> paramsBuffer;
  };

OIDN_NAMESPACE_END
