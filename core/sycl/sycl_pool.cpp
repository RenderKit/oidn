// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0#if defined(OIDN_DEVICE_SYCL)

#include "sycl_pool.h"

namespace oidn {

  struct SYCLPool
  {
    static constexpr int B = TensorAccessor3D<half, TensorLayout::Chw16c>::B;

    TensorAccessor3D<half, TensorLayout::Chw16c> src;
    TensorAccessor3D<half, TensorLayout::Chw16c> dst;

    OIDN_INLINE void operator ()(size_t hDst, size_t wDst) const SYCL_ESIMD_KERNEL
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

      simd<int16_t, B> v0, v1, v2, v3;
      v0.copy_from((int16_t*)srcPtr0);
      v1.copy_from((int16_t*)srcPtr1);
      v2.copy_from((int16_t*)srcPtr2);
      v3.copy_from((int16_t*)srcPtr3);

      // FIXME: use half
      simd<int16_t, B> v = esimd_max(esimd_max(v0, v1), esimd_max(v2, v3));
      v.copy_to((int16_t*)dstPtr);
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
    SYCLPool kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->executeESIMDKernel(dst->height() * dst->numChannelBlocks(), dst->width(), kernel);
  }

} // namespace oidn