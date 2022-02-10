// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0#if defined(OIDN_DEVICE_SYCL)

#include "sycl_pool.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct SYCLPool
  {
    static constexpr int B = TensorAccessor3D<T, layout>::B;

    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_INLINE void operator ()(size_t hDst, size_t wDst) const SYCL_ESIMD_FUNCTION
    { 
      using namespace sycl::ext::intel::experimental::esimd;

      const size_t hDstOffset = hDst * dst.hStride;
      const size_t wDstOffset = wDst * dst.wStride;
      
      const size_t dstOffset = hDstOffset     + wDstOffset;
      const size_t srcOffset = hDstOffset * 4 + wDstOffset * 2;

      char* srcPtr0 = src.ptr + srcOffset;
      char* srcPtr1 = srcPtr0 + src.wStride;
      char* srcPtr2 = srcPtr0 + src.hStride;
      char* srcPtr3 = srcPtr2 + src.wStride;
      char* dstPtr  = dst.ptr + dstOffset;

      const simd<T, B> v0 = block_load<T, B, vector_aligned_tag>((T*)srcPtr0);
      const simd<T, B> v1 = block_load<T, B, vector_aligned_tag>((T*)srcPtr1);
      const simd<T, B> v2 = block_load<T, B, vector_aligned_tag>((T*)srcPtr2);
      const simd<T, B> v3 = block_load<T, B, vector_aligned_tag>((T*)srcPtr3);

      const simd<T, B> v = max(max(v0, v1), max(v2, v3));
      block_store((T*)dstPtr, v);
    }
  };

  SYCLPoolNode::SYCLPoolNode(const Ref<SYCLDevice>& device, const PoolDesc& desc)
    : SYCLNode(device, desc.name),
      PoolNode(desc)
  {
    assert(src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());
  }

  void SYCLPoolNode::execute()
  {
    SYCLPool<half, TensorLayout::Chw16c> kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->executeESIMDKernel(dst->height() * dst->numChannelBlocks(), dst->width(), kernel);
  }

} // namespace oidn