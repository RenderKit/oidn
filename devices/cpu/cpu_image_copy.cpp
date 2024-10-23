// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_image_copy.h"
#include "cpu_image_copy_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUImageCopy::CPUImageCopy(CPUEngine* engine)
    : engine(engine)
  {}

  void CPUImageCopy::submitKernels(const Ref<CancellationToken>& ct)
  {
    check();

    ispc::CPUImageCopyKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    engine->submitFunc([=]
    {
      parallel_for(kernel.dst.H, [&](int h)
      {
        ispc::CPUImageCopyKernel_run(&kernel, h);
      });
    }, ct);
  }

OIDN_NAMESPACE_END