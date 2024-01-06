// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_process.h"
#include "cpu_output_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUOutputProcess::CPUOutputProcess(CPUEngine* engine, const OutputProcessDesc& desc)
    : OutputProcess(desc) {}

  void CPUOutputProcess::submit()
  {
    check();

    ispc::CPUOutputProcessKernel kernel;

    kernel.src = toISPC(*src);
    kernel.dst = toISPC(*dst);
    kernel.tile = toISPC(tile);
    kernel.transferFunc = toISPC(*transferFunc);
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    parallel_nd(kernel.tile.H, [&](int h)
    {
      ispc::CPUOutputProcessKernel_run(&kernel, h);
    });
  }

OIDN_NAMESPACE_END