// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_upsample.h"
#if defined(OIDN_DNNL)
  #include "cpu_upsample_ispc.h"
#endif

namespace oidn {

#if defined(OIDN_DNNL)

  CPUUpsample::CPUUpsample(const Ref<CPUDevice>& device, const UpsampleDesc& desc)
    : CPUOp(device),
      Upsample(desc) {}

  void CPUUpsample::run()
  {
    assert(src->getLayout() == TensorLayout::Chw8c ||
           src->getLayout() == TensorLayout::Chw16c);
    assert(src->getBlockSize() == device->getTensorBlockSize());

    ispc::CPUUpsampleKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    parallel_nd(src->getCB(), src->getH(), [&](int cb, int h)
    {
      ispc::CPUUpsampleKernel_run(&kernel, cb, h);
    });
  }

#else

  CPUUpsample::CPUUpsample(const Ref<CPUDevice>& device, const UpsampleDesc& desc)
    : CPUOp(device),
      Upsample(desc) {}

  void CPUUpsample::run()
  {
    assert(src->getLayout() == TensorLayout::chw);

    const size_t C = src->getC();
    const size_t H = src->getH();
    const size_t W = src->getW();

    parallel_nd(C, H, [&](int c, int h)
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

#endif

} // namespace oidn