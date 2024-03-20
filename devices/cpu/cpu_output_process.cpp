// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_process.h"
#include "cpu_output_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUOutputProcess::CPUOutputProcess(CPUEngine* engine, const OutputProcessDesc& desc)
    : OutputProcess(desc),
      engine(engine)
  {}

  void CPUOutputProcess::submit()
  {
    check();

    ispc::CPUOutputProcessKernel kernel;

    kernel.src = *src;
    kernel.dst = *dst;
    kernel.tile = toISPC(tile);
    kernel.transferFunc = toISPC(*transferFunc);
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    engine->submit([=]
    {
      parallel_for(kernel.tile.H, [&](int h)
      {
        ispc::CPUOutputProcessKernel_run(&kernel, h);
      });
    });
  }

OIDN_NAMESPACE_END