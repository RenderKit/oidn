// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_pool.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct SYCLPoolKernel
  {
    static constexpr int cBlock = TensorAccessor3D<T, layout>::cBlock;

    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_INLINE void operator ()(const WorkItem<2>& it) const SYCL_ESIMD_FUNCTION
    { 
      using namespace esimd;

      const size_t hDst = it.getId<0>();
      const size_t wDst = it.getId<1>();

      const size_t hDstOffset = hDst * dst.hStride;
      const size_t wDstOffset = wDst * dst.wStride;
      
      const size_t dstOffset = hDstOffset     + wDstOffset;
      const size_t srcOffset = hDstOffset * 4 + wDstOffset * 2;

      char* srcPtr0 = src.ptr + srcOffset;
      char* srcPtr1 = srcPtr0 + src.wStride;
      char* srcPtr2 = srcPtr0 + src.hStride;
      char* srcPtr3 = srcPtr2 + src.wStride;
      char* dstPtr  = dst.ptr + dstOffset;

      const simd<T, cBlock> v0 = block_load<T, cBlock, vector_aligned_tag>((T*)srcPtr0);
      const simd<T, cBlock> v1 = block_load<T, cBlock, vector_aligned_tag>((T*)srcPtr1);
      const simd<T, cBlock> v2 = block_load<T, cBlock, vector_aligned_tag>((T*)srcPtr2);
      const simd<T, cBlock> v3 = block_load<T, cBlock, vector_aligned_tag>((T*)srcPtr3);

      const simd<T, cBlock> v = max(max(v0, v1), max(v2, v3));
      block_store((T*)dstPtr, v);
    }
  };

  SYCLPool::SYCLPool(const Ref<SYCLDevice>& device, const PoolDesc& desc)
    : Pool(desc),
      device(device)
  {
    if (srcDesc.layout != TensorLayout::Chw16c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported pooling source layout/data type");
  }

  void SYCLPool::run()
  {
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    SYCLPoolKernel<half, TensorLayout::Chw16c> kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->runESIMDKernelAsync(WorkDim<2>(dst->getH() * dst->getCB(), dst->getW()), kernel);
  }

} // namespace oidn