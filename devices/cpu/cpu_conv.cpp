// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_conv.h"
#include "cpu_conv_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUConv::CPUConv(CPUEngine* engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {
    if ((srcDesc.layout != TensorLayout::Chw8c &&
         srcDesc.layout != TensorLayout::Chw16c) || srcDesc.dataType != DataType::Float32)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
      throw std::invalid_argument("unsupported convolution kernel size");
    if ((weightDesc.layout != TensorLayout::IOhw8i8o &&
         weightDesc.layout != TensorLayout::IOhw16i16o) || weightDesc.dataType != DataType::Float32)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float32)
      throw std::invalid_argument("unsupported convolution bias layout/data type");

    const int blockC = getTensorLayoutInfo(dstDesc.layout).blockC;
    const int IC = srcDesc.getPaddedC();
    const int OC = dstDesc.getPaddedC();
    const int OH = dstDesc.getH();
    const int OW = dstDesc.getW();

    const int OCB = OC / blockC;
    blockOCB = min(OCB, ispc::CPUConvKernel_getMaxBlockOCB());
    while (OCB % blockOCB != 0)
      blockOCB--;

    OCBB = OCB / blockOCB;
    blockOW = ispc::CPUConvKernel_getBlockOW(blockOCB);

    // Split the output width into tiles to fit into the L2 cache
    const size_t cacheSize = 512 * 1024; // FIXME: query actual size but this also works well

    const int tileOW = max(
      static_cast<int>((cacheSize - weightDesc.getByteSize() - biasDesc.getByteSize()) /
                       (IC * getDataTypeSize(srcDesc.dataType)  * weightDesc.getH() +
                        OC * getDataTypeSize(dstDesc.dataType)) - weightDesc.getW() + 1),
      1);

    const int maxOWT = max(OW / (2*blockOW), 1); // max number of OW tiles
    OWT = min(ceil_div(OW, tileOW), maxOWT);     // number of OW tiles

    // Tweak the number of OW tiles to maximize threading efficiency
    const int numThreads = engine->getNumThreads();
    double bestThreadEff = 0;
    for (int curOWT = OWT; curOWT < maxOWT; ++curOWT)
    {
      const size_t N = size_t(OCBB) * OH * curOWT; // work amount
      const double threadEff = 1. - double(N % numThreads) / N;
      if (threadEff > bestThreadEff)
      {
        OWT = curOWT;
        bestThreadEff = threadEff;
        if (threadEff >= 0.99)
          break;
      }
    }
  }

  void CPUConv::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src || !dst)
      throw std::logic_error("conving source/destination not set");

    ispc::CPUConvKernel kernel;
    kernel.src    = *src;
    kernel.weight = *weight;
    kernel.bias   = *bias;
    kernel.dst    = *dst;
    kernel.relu   = activation == Activation::ReLU;

    engine->submitFunc([=]
    {
      const int OH = kernel.dst.H;
      const int OW = kernel.dst.W;
      const size_t N = size_t(OCBB) * OH * OWT;

      parallel_for(N, [&](size_t i)
      {
        const size_t j = i / OCBB;
        const int ocbb = int(i % OCBB);
        const int oh   = int(j % OH);
        const int owt  = int(j / OH);

        constexpr int PW = 1; // KW = 3
        const int owr = OWT * (blockOW - PW - 1);
        const int owBegin = owt   > 0   ? (owt     * OW + owr) / (OWT*blockOW) * blockOW + PW : 0;
        const int owEnd   = owt+1 < OWT ? ((owt+1) * OW + owr) / (OWT*blockOW) * blockOW + PW : OW;

        ispc::CPUConvKernel_run(&kernel, blockOCB, ocbb * blockOCB, oh, owBegin, owEnd);
      });
    }, ct);
  }

OIDN_NAMESPACE_END