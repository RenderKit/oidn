// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_input_process.h"
#include "cpu_input_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUInputProcess::CPUInputProcess(CPUEngine* engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc),
      engine(engine)
  {}

  void CPUInputProcess::submitKernels(const Ref<CancellationToken>& ct)
  {
    check();

    ispc::CPUInputProcessKernel kernel;
    Image nullImage;

    kernel.input  = color ? *color : (albedo ? *albedo : *normal);
    kernel.albedo = (color && albedo) ? *albedo : nullImage;
    kernel.normal = (color && normal) ? *normal : nullImage;
    kernel.dst    = *dst;
    kernel.tile   = toISPC(tile);
    kernel.transferFunc = toISPC(*transferFunc);
    kernel.hdr   = hdr;
    kernel.snorm = snorm;

    engine->submitFunc([=]
    {
      parallel_for(kernel.dst.H, [&](int hDst)
      {
        ispc::CPUInputProcessKernel_run(&kernel, hDst);
      });
    }, ct);
  }

OIDN_NAMESPACE_END