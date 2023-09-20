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
    if (srcDesc.layout != TensorLayout::Chw8c &&
        srcDesc.layout != TensorLayout::Chw16c &&
        srcDesc.layout != TensorLayout::chw)
      throw std::invalid_argument("unsupported pooling source layout");
  }

  void CPUPool::submit()
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    if (srcDesc.layout == TensorLayout::Chw8c ||
        srcDesc.layout == TensorLayout::Chw16c)
    {

      ispc::CPUPoolKernel kernel;
      kernel.src = toISPC<ispc::TensorAccessor3D>(*src);
      kernel.dst = toISPC<ispc::TensorAccessor3D>(*dst);

      // Blocked layout
      const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;

      parallel_nd(dst->getPaddedC() / blockC, dst->getH(), [&](int cb, int h)
      {
        ispc::CPUPoolKernel_run_blocked(&kernel, cb, h);
      });
    }
    else // CHW
    {
      TensorAccessor3D<float, TensorLayout::chw> srcAccessor = *src;
      TensorAccessor3D<float, TensorLayout::chw> dstAccessor = *dst;

      parallel_nd(dst->getPaddedC(), dst->getH(), [&](int c, int h)
      {
        for(int w = 0; w < dst->getW(); w++)
        {
          const float x0 = srcAccessor(c, h*2,   w*2);
          const float x1 = srcAccessor(c, h*2,   w*2+1);
          const float x2 = srcAccessor(c, h*2+1, w*2);
          const float x3 = srcAccessor(c, h*2+1, w*2+1);

          dstAccessor(c, h, w) = math::max(math::max(x0, x1), math::max(x2, x3));
        }
      });
    }
  }

OIDN_NAMESPACE_END