// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_upsample.h"
#include "cpu_upsample_ispc.h"
#include "cpu_common.h"

namespace oidn {

  CPUUpsample::CPUUpsample(const Ref<CPUEngine>& engine, const UpsampleDesc& desc)
    : Upsample(desc),
      engine(engine)
  {
    if (srcDesc.layout != TensorLayout::chw &&
        srcDesc.layout != TensorLayout::Chw8c &&
        srcDesc.layout != TensorLayout::Chw16c)
      throw std::invalid_argument("unsupported upsampling source layout");
  }

  void CPUUpsample::submit()
  {
    if (!src || !dst)
      throw std::logic_error("upsampling source/destination not set");

    if (srcDesc.layout != TensorLayout::chw)
    {
      const int blockC = getTensorLayoutBlockC(srcDesc.layout);

      ispc::CPUUpsampleKernel kernel;
      kernel.src = toISPC(*src);
      kernel.dst = toISPC(*dst);

      parallel_nd(src->getC() / blockC, src->getH(), [&](int cb, int h)
      {
        ispc::CPUUpsampleKernel_run(&kernel, cb, h);
      });
    }
    else
    {
      const size_t H = src->getH();
      const size_t W = src->getW();

      parallel_nd(src->getC(), src->getH(), [&](int c, int h)
      {
        const size_t offset = (c*H + h) * W;
        const float* srcPtr_line = (float*)src->getData() + offset;
        float* dstPtr_line0 = (float*)dst->getData() + offset * 4;
        float* dstPtr_line1 = dstPtr_line0 + W*2; // next line

        #pragma unroll(16)
        for (size_t w = 0; w < W; ++w)
        {
          // Load value
          const float value = srcPtr_line[w];

          // Store value 2x2
          dstPtr_line0[w*2  ] = value;
          dstPtr_line0[w*2+1] = value;
          dstPtr_line1[w*2  ] = value;
          dstPtr_line1[w*2+1] = value;
        }
      });
    }
  }
} // namespace oidn