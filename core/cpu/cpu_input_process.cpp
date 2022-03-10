// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_input_process.h"
#include "input_process_kernel_ispc.h"

namespace oidn {

  CPUInputProcess::CPUInputProcess(const Ref<CPUDevice>& device, const InputProcessDesc& desc)
    : CPUOp(device),
      InputProcess(desc) {}

  void CPUInputProcess::run()
  {
    assert(tile.H + tile.hSrcBegin <= getInput()->getH());
    assert(tile.W + tile.wSrcBegin <= getInput()->getW());
    assert(tile.H + tile.hDstBegin <= dst->getH());
    assert(tile.W + tile.wDstBegin <= dst->getW());

    ispc::InputProcessKernel kernel;

    kernel.color  = color  ? *color  : Image();
    kernel.albedo = albedo ? *albedo : Image();
    kernel.normal = normal ? *normal : Image();
    kernel.dst = *dst;
    kernel.tile = tile;
    kernel.transferFunc = *transferFunc;
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    parallel_nd(kernel.dst.H, [&](int hDst)
    {
      ispc::InputProcessKernel_run(&kernel, hDst);
    });
  }

} // namespace oidn