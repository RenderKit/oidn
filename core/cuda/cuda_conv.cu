// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_conv.h"
#include "cutlass_conv.h"

namespace oidn {

  std::shared_ptr<Conv> newCUDAConv(const Ref<CUDADevice>& device, const ConvDesc& desc)
  {
    // Get the list of kernels supported by the device
    std::vector<CutlassConvFactory> kernels;
    const int sm = device->getComputeCapability();
    if (sm >= 80)
      kernels = getCutlassConvInstances<80>();
    else if (sm >= 75)
      kernels = getCutlassConvInstances<75>();
    else if (sm >= 70)
      kernels = getCutlassConvInstances<70>();
    else
      throw std::runtime_error("could not find a supported convolution kernel");

    // Select the likely fastest compatible kernel
    const auto problemSize = toCutlassProblemSize(desc);
    const auto gemmSize = cutlass::conv::implicit_gemm_problem_size(cutlass::conv::Operator::kFprop, problemSize);
    const size_t M = gemmSize.m();
    const size_t N = gemmSize.n();
    const size_t K = gemmSize.k();

    const CutlassConvFactory* bestKernel = nullptr;
    int bestBlockSize = 0;
    size_t bestCost = std::numeric_limits<size_t>::max();

    for (const auto& kernel : kernels)
    {
      if (kernel.dataType != desc.srcDesc.dataType)
        continue;

      const int blockSize = kernel.blockM * kernel.blockN * kernel.blockK;
      const size_t cost = round_up(M, kernel.blockM) * round_up(N, kernel.blockN) * round_up(K, kernel.blockK);

      if ((cost < bestCost) || (cost == bestCost && blockSize > bestBlockSize))
      {
        bestKernel = &kernel;
        bestBlockSize = blockSize;
        bestCost = cost;
      }
    }

    if (!bestKernel)
      throw std::runtime_error("could not find a compatible convolution kernel");

    return bestKernel->make(device, desc);
  }

} // namespace oidn