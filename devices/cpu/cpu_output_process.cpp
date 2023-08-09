// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_process.h"
#include "cpu_output_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUOutputProcess::CPUOutputProcess(const Ref<CPUEngine>& engine, const OutputProcessDesc& desc)
    : OutputProcess(desc),
      engine(engine) {}

  void CPUOutputProcess::submit()
  {
    if (!src || !dst)
      throw std::logic_error("output processing source/destination not set");
    if (tile.hSrcBegin + tile.H > src->getH() ||
        tile.wSrcBegin + tile.W > src->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("output processing source/destination out of range");

    ispc::CPUOutputProcessKernel kernel;

    kernel.src = toISPC<ispc::TensorAccessor3D>(*src);
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