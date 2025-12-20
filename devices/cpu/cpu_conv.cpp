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
    const int IC = srcDesc.getPaddedC(); // input channels
    const int OC = dstDesc.getPaddedC(); // output channels
    const int OH = dstDesc.getH();       // output height
    const int OW = dstDesc.getW();       // output width

    const int OCB = OC / blockC; // number of output channel blocks
    blockOCB = min(OCB, ispc::CPUConvKernel_getMaxBlockOCB());
    while (OCB % blockOCB != 0)
      blockOCB--;

    OCBB = OCB / blockOCB; // number of output channel block blocks
    blockOW = ispc::CPUConvKernel_getBlockOW(blockOCB);

    // Split the output width into chunks to fit into the L2 cache
    const size_t cacheSize = 512 * 1024; // FIXME: query actual size but this also works well

    const int chunkOW = max(
      static_cast<int>((cacheSize - weightDesc.getByteSize() - biasDesc.getByteSize()) /
                       (IC * getDataTypeSize(srcDesc.dataType)  * weightDesc.getH() +
                        OC * getDataTypeSize(dstDesc.dataType)) - weightDesc.getW() + 1),
      1);

    const int maxOWC = max(OW / (2*blockOW), 1); // max number of output width chunks
    OWC = min(ceil_div(OW, chunkOW), maxOWC);    // number of output width chunks

    // Tweak the number of output width chunks to maximize threading efficiency
    const int numThreads = engine->getNumThreads();
    double bestThreadEff = 0;
    for (int curOWC = OWC; curOWC < maxOWC; ++curOWC)
    {
      const size_t N = size_t(OCBB) * OH * curOWC; // work amount
      const double threadEff = 1. - double(N % numThreads) / N;
      if (threadEff > bestThreadEff)
      {
        OWC = curOWC;
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
      const int OH = kernel.dst.H; // output height
      const int OW = kernel.dst.W; // output width
      const size_t N = size_t(OCBB) * OH * OWC; // total number of work items

      parallel_for(N, [&](size_t i)
      {
        const size_t j = i / OCBB;
        const int ocbb = int(i % OCBB); // output channel block block index
        const int oh   = int(j % OH);   // output height index
        const int owc  = int(j / OH);   // output width chunk index

        constexpr int PW = 1; // KW = 3
        const int owr = OWC * (blockOW - PW - 1);
        const int owBegin = owc   > 0   ? (owc     * OW + owr) / (OWC*blockOW) * blockOW + PW : 0;
        const int owEnd   = owc+1 < OWC ? ((owc+1) * OW + owr) / (OWC*blockOW) * blockOW + PW : OW;

        ispc::CPUConvKernel_run_f32(&kernel, blockOCB, ocbb * blockOCB, oh, owBegin, owEnd);
      });
    }, ct);
  }

OIDN_NAMESPACE_END