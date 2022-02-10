// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_upsample.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct SYCLUpsample
  {
    static constexpr int B = TensorAccessor3D<T, layout>::B;

    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_INLINE void operator ()(size_t hSrc, size_t wSrc) const SYCL_ESIMD_FUNCTION
    {
      using namespace sycl::ext::intel::experimental::esimd;

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

  SYCLUpsampleNode::SYCLUpsampleNode(const Ref<SYCLDevice>& device, const UpsampleDesc& desc)
    : SYCLNode(device, desc.name),
      UpsampleNode(desc)
  {
    assert(src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());
  }

  void SYCLUpsampleNode::execute()
  {
    SYCLUpsample<half, TensorLayout::Chw16c> kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    device->executeESIMDKernel(src->height() * src->numChannelBlocks(), src->width(), kernel);
  }

} // namespace oidn