// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_upsample.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct SYCLUpsampleKernel
  {
    static constexpr int B = TensorAccessor3D<T, layout>::B;

    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_INLINE void operator ()(const WorkItem<2>& it) const SYCL_ESIMD_FUNCTION
    {
      using namespace sycl::ext::intel::esimd;

      const size_t hSrc = it.getId<0>();
      const size_t wSrc = it.getId<1>();

      const size_t hSrcOffset = hSrc * src.hStride;
      const size_t wSrcOffset = wSrc * src.wStride;
      
      const size_t srcOffset = hSrcOffset     + wSrcOffset;
      const size_t dstOffset = hSrcOffset * 4 + wSrcOffset * 2;

      char* srcPtr  = src.ptr + srcOffset;
      char* dstPtr0 = dst.ptr + dstOffset;
      char* dstPtr2 = dstPtr0 + dst.hStride;

      const simd<T, B> v = block_load<T, B, vector_aligned_tag>((T*)srcPtr);

      const simd<T, B*2> v2 = v.template replicate<2>();
      block_store((T*)dstPtr0, v2);
      block_store((T*)dstPtr2, v2);
    }
  };

  SYCLUpsample::SYCLUpsample(const Ref<SYCLDevice>& device, const UpsampleDesc& desc)
    : Upsample(desc),
      device(device)
  {
    assert(src->getLayout() == TensorLayout::Chw16c);
    assert(src->getBlockSize() == device->getTensorBlockSize());
  }

  void SYCLUpsample::run()
  {
    SYCLUpsampleKernel<half, TensorLayout::Chw16c> kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->runESIMDKernelAsync({src->getH() * src->getCB(), src->getW()}, kernel);
  }

} // namespace oidn