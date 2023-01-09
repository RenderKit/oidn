// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_pool.h"
#include "cpu_pool_ispc.h"
#include "cpu_common.h"

namespace oidn {

  CPUPool::CPUPool(const Ref<CPUEngine>& engine, const PoolDesc& desc)
    : Pool(desc),
      engine(engine)
  {
    if (srcDesc.layout != TensorLayout::Chw8c &&
        srcDesc.layout != TensorLayout::Chw16c)
      throw std::invalid_argument("unsupported pooling source layout");
  }

  void CPUPool::submit()
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    const int blockC = getTensorLayoutBlockC(dstDesc.layout);

    ispc::CPUPoolKernel kernel;
    kernel.src = toISPC(*src);
    kernel.dst = toISPC(*dst);

    parallel_nd(dst->getC() / blockC, dst->getH(), [&](int cb, int h)
    {
      ispc::CPUPoolKernel_run(&kernel, cb, h);
    });
  }

} // namespace oidn