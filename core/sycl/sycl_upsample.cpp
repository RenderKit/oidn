// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_upsample.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct SYCLUpsampleKernel
  {
    static constexpr int cBlock = TensorAccessor3D<T, layout>::cBlock;

    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_INLINE void operator ()(const WorkItem<2>& it) const SYCL_ESIMD_FUNCTION
    {
      using namespace esimd;

      const size_t hSrc = it.getId<0>();
      const size_t wSrc = it.getId<1>();

      const size_t hSrcOffset = hSrc * src.hStride;
      const size_t wSrcOffset = wSrc * src.wStride;
      
      const size_t srcOffset = hSrcOffset     + wSrcOffset;
      const size_t dstOffset = hSrcOffset * 4 + wSrcOffset * 2;

      char* srcPtr  = src.ptr + srcOffset;
      char* dstPtr0 = dst.ptr + dstOffset;
      char* dstPtr2 = dstPtr0 + dst.hStride;

      const simd<T, cBlock> v = block_load<T, cBlock, vector_aligned_tag>((T*)srcPtr);

      const simd<T, cBlock*2> v2 = v.template replicate<2>();
      block_store((T*)dstPtr0, v2);
      block_store((T*)dstPtr2, v2);
    }
  };

  SYCLUpsample::SYCLUpsample(const Ref<SYCLDevice>& device, const UpsampleDesc& desc)
    : Upsample(desc),
      device(device)
  {
    if (srcDesc.layout != TensorLayout::Chw16c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported upsampling source layout/data type");
  }

  void SYCLUpsample::run()
  {
    if (!src || !dst)
      throw std::logic_error("upsampling source/destination not set");

    SYCLUpsampleKernel<half, TensorLayout::Chw16c> kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->runESIMDKernelAsync(WorkDim<2>(src->getH() * src->getCB(), src->getW()), kernel);
  }

} // namespace oidn