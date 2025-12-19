// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_conv_amx.h"
#include "cpu_common.h"
#include "cpu_conv_amx_ispc.h"

OIDN_NAMESPACE_BEGIN

  CPUConvAMX::CPUConvAMX(CPUEngine* engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {
    if (srcDesc.layout != TensorLayout::Chw32c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
      throw std::invalid_argument("unsupported convolution kernel size");
    if (weightDesc.layout != TensorLayout::OIhw2o16i16o2i || weightDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution bias layout/data type");
  }

  void CPUConvAMX::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src || !dst)
      throw std::logic_error("conving source/destination not set");

    ispc::CPUConvAMXKernel kernel;
    ispc::CPUConvAMXKernel_init(&kernel);

    kernel.src    = *src;
    kernel.weight = *weight;
    kernel.bias   = *bias;
    kernel.dst    = *dst;
    kernel.relu   = activation == Activation::ReLU;

    switch (postOp)
    {
    case PostOp::None:     kernel.postOp = ispc::PostOp_None;     break;
    case PostOp::Pool:     kernel.postOp = ispc::PostOp_Pool;     break;
    case PostOp::Upsample: kernel.postOp = ispc::PostOp_Upsample; break;
    default:
      throw std::logic_error("unsupported convolution postop");
    }

    // Block sizes of the kernel
    constexpr int blockOC = 32; // output channel block size
    constexpr int blockOW = 16; // output block width before post-op
    constexpr int blockOH = 2;  // output block height before post-op

    constexpr int chunkOH = 8; // we group output height blocks into larger chunks

    engine->submitFunc([=]
    {
      const int OC = kernel.dst.C; // output channels
      const int OH = kernel.src.H; // output height before post-op
      const int OW = kernel.src.W; // output width before post-op

      const int OCB = OC / blockOC;          // number of output channel blocks
      const int OHC = ceil_div(OH, chunkOH); // number of output height chunks
      const int OWB = ceil_div(OW, blockOW); // number of output width blocks

      const size_t N = size_t(OCB) * OHC * OWB; // total number of work items

      tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [&](const tbb::blocked_range<size_t>& r)
      {
        for (size_t i = r.begin(); i != r.end(); ++i)
        {
          const size_t j = i / OCB;
          const int ocb = int(i % OCB); // output channel block index
          const int owb = int(j % OWB); // output width block index
          const int ohc = int(j / OWB); // output height chunk index

          const int oc = ocb * blockOC;
          const int ow = owb * blockOW;
          const int ohBegin = round_up(ohc * OH / OHC, blockOH);
          const int ohEnd   = min(round_up((ohc + 1) * OH / OHC, blockOH), OH);

          const bool isFirst = (i == r.begin());
          const bool isLast  = (i == r.end() - 1);

          ispc::CPUConvAMXKernel_run_f16(&kernel, oc, ohBegin, ohEnd, ow, isFirst, isLast);
        }
      });
    }, ct);
  }

OIDN_NAMESPACE_END