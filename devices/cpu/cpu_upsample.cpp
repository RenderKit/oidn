// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_upsample.h"
#include "cpu_upsample_f32_ispc.h"
#include "cpu_upsample_f16_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUUpsample::CPUUpsample(CPUEngine* engine, const UpsampleDesc& desc)
    : Upsample(desc),
      engine(engine)
  {
    if (srcDesc.layout != TensorLayout::chw    &&
        srcDesc.layout != TensorLayout::Chw8c  &&
        srcDesc.layout != TensorLayout::Chw16c &&
        srcDesc.layout != TensorLayout::Chw32c)
      throw std::invalid_argument("unsupported upsampling source layout");
    if (srcDesc.dataType != DataType::Float32 &&
        (srcDesc.dataType != DataType::Float16 || srcDesc.layout == TensorLayout::chw))
      throw std::invalid_argument("unsupported upsampling source data type");
  }

  void CPUUpsample::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src || !dst)
      throw std::logic_error("upsampling source/destination not set");

    if (srcDesc.layout != TensorLayout::chw)
    {
      const int blockC = getTensorLayoutInfo(srcDesc.layout).blockC;

      ispc::CPUUpsampleKernel kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      auto kernelFunc = (src->getDataType() == DataType::Float16)
        ? ispc::CPUUpsampleKernel_run_f16
        : ispc::CPUUpsampleKernel_run_f32;

      engine->submitFunc([=]
      {
        parallel_for(kernel.src.C / blockC, kernel.src.H, [&](int cb, int h)
        {
          kernelFunc(&kernel, cb, h);
        });
      }, ct);
    }
    else
    {
      const int C = src->getPaddedC();
      const size_t H = src->getH();
      const size_t W = src->getW();
      const float* srcPtr = (float*)src->getPtr();
      float* dstPtr = (float*)dst->getPtr();

      engine->submitFunc([=]
      {
        parallel_for(C, H, [&](int c, size_t h)
        {
          const size_t offset = (c*H + h) * W;
          const float* srcPtr_line = srcPtr + offset;
          float* dstPtr_line0 = dstPtr + offset * 4;
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
      }, ct);
    }
  }
OIDN_NAMESPACE_END