// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_input_process.h"
#include "cpu_input_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUInputProcess::CPUInputProcess(CPUEngine* engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc) {}

  void CPUInputProcess::submit()
  {
    check();

    ispc::CPUInputProcessKernel kernel;
    Image nullImage;

    kernel.input  = toISPC(color ? *color : (albedo ? *albedo : *normal));
    kernel.albedo = toISPC((color && albedo) ? *albedo : nullImage);
    kernel.normal = toISPC((color && normal) ? *normal : nullImage);
    kernel.dst    = toISPC(*dst);
    kernel.tile   = toISPC(tile);
    kernel.transferFunc = toISPC(*transferFunc);
    kernel.hdr   = hdr;
    kernel.snorm = snorm;

    parallel_nd(kernel.dst.H, [&](int hDst)
    {
      ispc::CPUInputProcessKernel_run(&kernel, hDst);
    });
  }

OIDN_NAMESPACE_END