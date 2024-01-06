// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_conv.h"
#include "cutlass_conv.h"

OIDN_NAMESPACE_BEGIN

  Ref<Conv> newCUDAConv(CUDAEngine* engine, const ConvDesc& desc)
  {
    // Get the list of kernels supported by the engine
    std::vector<CutlassConvFactory> kernels;
    const int smArch = engine->getSMArch();
    if (smArch >= 80)
      kernels = getCutlassConvInstances<80>();
    else if (smArch >= 75)
      kernels = getCutlassConvInstances<75>();
    else if (smArch >= 70)
      kernels = getCutlassConvInstances<70>();
    else
      throw std::runtime_error("unsupported convolution");

    // Select the likely fastest compatible kernel
    const auto problemSize = toCutlassProblemSize(desc);
    const auto gemmSize = cutlass::conv::implicit_gemm_problem_size(cutlass::conv::Operator::kFprop, problemSize);
    const size_t M = gemmSize.m();
    const size_t N = gemmSize.n();
    const size_t K = gemmSize.k();

    const DataType accumType = desc.fastMath ? desc.srcDesc.dataType : DataType::Float32;

    const CutlassConvFactory* bestKernel = nullptr;
    int bestBlockSize = 0;
    size_t bestCost = std::numeric_limits<size_t>::max();

    for (const auto& kernel : kernels)
    {
      if (kernel.dataType != desc.srcDesc.dataType || kernel.accumType < accumType)
        continue;

      const int blockSize = kernel.blockM * kernel.blockN * kernel.blockK;
      const size_t cost = round_up(M, kernel.blockM) * round_up(N, kernel.blockN) * round_up(K, kernel.blockK);

      if ((cost < bestCost) ||
          (cost == bestCost && blockSize > bestBlockSize) ||
          (cost == bestCost && blockSize == bestBlockSize && kernel.accumType == accumType))
      {
        bestKernel = &kernel;
        bestBlockSize = blockSize;
        bestCost = cost;
      }
    }

    if (!bestKernel)
      throw std::runtime_error("unsupported convolution");

    return bestKernel->make(engine, desc);
  }

OIDN_NAMESPACE_END