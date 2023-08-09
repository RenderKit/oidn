// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_pool.h"
#include "cpu_pool_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUPool::CPUPool(const Ref<CPUEngine>& engine, const PoolDesc& desc)
    : Pool(desc),
      engine(engine)
  {
    /*if (srcDesc.layout != TensorLayout::Chw8c &&
        srcDesc.layout != TensorLayout::Chw16c)
      throw std::invalid_argument("unsupported pooling source layout");
  */}

  void CPUPool::submit()
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    ispc::CPUPoolKernel kernel;
    kernel.src = toISPC<ispc::TensorAccessor3D>(*src);
    kernel.dst = toISPC<ispc::TensorAccessor3D>(*dst);

    // Blocked layout
    if(src->getLayout() == TensorLayout::Chw8c || src->getLayout() == TensorLayout::Chw16c)
    {
      const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;

      parallel_nd(dst->getPaddedC() / blockC, dst->getH(), [&](int cb, int h)
      {
        ispc::CPUPoolKernel_run_blocked(&kernel, cb, h);
      });
    }
    else // Non-blocked layout
    {
      parallel_nd(dst->getPaddedC(), dst->getH(), dst->getW(), [&](int c, int h, int w)
      {
        ispc::CPUPoolKernel_run_nonblocked(&kernel, c, h, w);
      });
    }
  }

OIDN_NAMESPACE_END