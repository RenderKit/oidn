// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_upsample.h"

namespace oidn {

  struct SYCLUpsampleKernel
  {
    static constexpr int B = TensorAccessor3D<half, TensorLayout::Chw16c>::B;

    TensorAccessor3D<half, TensorLayout::Chw16c> src;
    TensorAccessor3D<half, TensorLayout::Chw16c> dst;

    OIDN_INLINE void operator ()(size_t hSrc, size_t wSrc) const SYCL_ESIMD_KERNEL
    { 
      using namespace sycl::ext::intel::experimental::esimd;

      const size_t hSrcOffset = hSrc * src.hStride;
      const size_t wSrcOffset = wSrc * src.wStride;
      
      const size_t srcOffset = hSrcOffset     + wSrcOffset;
      const size_t dstOffset = hSrcOffset * 4 + wSrcOffset * 2;

      char* srcPtr  = src.ptr + srcOffset;
      char* dstPtr0 = dst.ptr + dstOffset;
      char* dstPtr2 = dstPtr0 + dst.hStride;

      simd<int16_t, B> v;
      v.copy_from((int16_t*)srcPtr);

      simd<int16_t, B*2> v2 = v.replicate<2, 0, B, 1>(0);
      v2.copy_to((int16_t*)dstPtr0);
      v2.copy_to((int16_t*)dstPtr2);
    }
  };

  SYCLUpsample::SYCLUpsample(const Ref<SYCLDevice>& device, const UpsampleDesc& desc)
    : SYCLOp(device),
      Upsample(desc)
  {
    assert(src->getLayout() == TensorLayout::Chw16c);
    assert(src->getBlockSize() == device->getTensorBlockSize());
  }

  void SYCLUpsample::run()
  {
    SYCLUpsampleKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->runESIMDKernel(src->getH() * src->getCB(), src->getW(), kernel);
  }

} // namespace oidn