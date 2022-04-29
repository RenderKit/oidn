// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_process.h"
#include "cpu_output_process_ispc.h"

namespace oidn {

  CPUOutputProcess::CPUOutputProcess(const Ref<CPUDevice>& device, const OutputProcessDesc& desc)
    : OutputProcess(desc),
      device(device) {}

  void CPUOutputProcess::run()
  {
    if (!src || !dst)
      throw std::logic_error("output processing source/destination not set");
    if (tile.hSrcBegin + tile.H > src->getH() ||
        tile.wSrcBegin + tile.W > src->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("output processing source/destination out of range");

    ispc::CPUOutputProcessKernel kernel;

    kernel.src = *src;
    kernel.dst = *dst;
    kernel.tile = tile;
    kernel.transferFunc = *transferFunc;
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    parallel_nd(kernel.tile.H, [&](int h)
    {
      ispc::CPUOutputProcessKernel_run(&kernel, h);
    });
  }

} // namespace oidn