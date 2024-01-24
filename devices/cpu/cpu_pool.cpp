// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_pool.h"
#include "cpu_pool_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUPool::CPUPool(CPUEngine* engine, const PoolDesc& desc)
    : Pool(desc)
  {
    if (srcDesc.layout != TensorLayout::Chw8c &&
        srcDesc.layout != TensorLayout::Chw16c)
      throw std::invalid_argument("unsupported pooling source layout");
  }

  void CPUPool::submit()
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;

    ispc::CPUPoolKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    parallel_nd(dst->getPaddedC() / blockC, dst->getH(), [&](int cb, int h)
    {
      ispc::CPUPoolKernel_run(&kernel, cb, h);
    });
  }

OIDN_NAMESPACE_END