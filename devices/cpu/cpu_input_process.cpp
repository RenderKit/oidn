// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_input_process.h"
#include "cpu_input_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUInputProcess::CPUInputProcess(const Ref<CPUEngine>& engine, const InputProcessDesc& desc)
    : InputProcess(engine, desc),
      engine(engine) {}

  void CPUInputProcess::submit()
  {
    if (!getMainSrc() || !dst)
      throw std::logic_error("input processing source/destination not set");
    if (tile.hSrcBegin + tile.H > getMainSrc()->getH() ||
        tile.wSrcBegin + tile.W > getMainSrc()->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("input processing source/destination out of range");

    ispc::CPUInputProcessKernel kernel;
    Image nullImage;

    kernel.color  = toISPC(color  ? *color  : nullImage);
    kernel.albedo = toISPC(albedo ? *albedo : nullImage);
    kernel.normal = toISPC(normal ? *normal : nullImage);
    kernel.dst    = toISPC<ispc::TensorAccessor3D>(*dst);
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