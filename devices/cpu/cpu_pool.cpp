// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_pool.h"
#include "cpu_pool_f32_ispc.h"
#include "cpu_pool_f16_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUPool::CPUPool(CPUEngine* engine, const PoolDesc& desc)
    : Pool(desc),
      engine(engine)
  {
    if (srcDesc.layout != TensorLayout::Chw8c  &&
        srcDesc.layout != TensorLayout::Chw16c &&
        srcDesc.layout != TensorLayout::Chw32c)
      throw std::invalid_argument("unsupported pooling source layout");
    if (srcDesc.dataType != DataType::Float32 && srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported pooling source data type");
  }

  void CPUPool::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;

    ispc::CPUPoolKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    auto kernelFunc = (src->getDataType() == DataType::Float16)
      ? ispc::CPUPoolKernel_run_f16
      : ispc::CPUPoolKernel_run_f32;

    engine->submitFunc([=]
    {
      parallel_for(kernel.dst.C / blockC, kernel.dst.H, [&](int cb, int h)
      {
        kernelFunc(&kernel, cb, h);
      });
    }, ct);
  }

OIDN_NAMESPACE_END