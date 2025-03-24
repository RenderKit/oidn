// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_conv.h"
#include "ck_conv.h"

OIDN_NAMESPACE_BEGIN

  Ref<Conv> newHIPConv(HIPEngine* engine, const ConvDesc& desc)
  {
    // Get the list of kernels supported by the engine
    std::vector<CKConvFactory> kernels;
    switch (engine->getArch())
    {
    case HIPArch::DL:
      kernels = getCKConvInstances<HIPArch::DL>(desc.activation);
      break;
    case HIPArch::WMMA:
      kernels = getCKConvInstances<HIPArch::WMMA>(desc.activation);
      break;
    default:
      throw std::runtime_error("unsupported architecture");
    }

    // Select the likely fastest compatible kernel based on the GEMM dimensions
    const size_t M = desc.srcDesc.getH() * desc.srcDesc.getW(); // == destination dims
    const size_t N = desc.weightDesc.getPaddedO();
    const size_t K = desc.weightDesc.getPaddedI() * desc.weightDesc.getH() * desc.weightDesc.getW();

    const DataType accumType = DataType::Float32;

    const CKConvFactory* bestKernel = nullptr;
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