// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(OIDN_DEVICE_GPU)
  #include "sycl_device.h"
#endif

#include "pool.h"

namespace oidn {

#if defined(OIDN_DEVICE_GPU)

  struct Pool
  {
    static constexpr int K = TensorAccessor<half>::K;

    TensorAccessor<half> src;
    TensorAccessor<half> dst;

    __forceinline void operator ()(size_t hDst, size_t wDst) const SYCL_ESIMD_KERNEL
    { 
      using namespace sycl::ext::intel::experimental::esimd;

      const size_t hDstOffset = hDst * dst.rowStride;
      const size_t wDstOffset = wDst * dst.itemStride;
      
      const size_t dstOffset = hDstOffset     + wDstOffset;
      const size_t srcOffset = hDstOffset * 4 + wDstOffset * 2;

      char* srcPtr0 = src.ptr + srcOffset;
      char* srcPtr1 = srcPtr0 + src.itemStride;
      char* srcPtr2 = srcPtr0 + src.rowStride;
      char* srcPtr3 = srcPtr2 + src.itemStride;
      char* dstPtr  = dst.ptr + dstOffset;

      simd<int16_t, K> v0, v1, v2, v3;
      v0.copy_from((int16_t*)srcPtr0);
      v1.copy_from((int16_t*)srcPtr1);
      v2.copy_from((int16_t*)srcPtr2);
      v3.copy_from((int16_t*)srcPtr3);

      // FIXME: use half
      simd<int16_t, K> v = esimd_max(esimd_max(v0, v1), esimd_max(v2, v3));
      v.copy_to((int16_t*)dstPtr);
    }
  };

  SYCLPoolNode::SYCLPoolNode(const Ref<SYCLDevice>& device,
                             const std::string& name,
                             const std::shared_ptr<Tensor>& src,
                             const std::shared_ptr<Tensor>& dst)
    : Node(device, name),
      src(src),
      dst(dst)
  {
    assert(src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());
  }

  void SYCLPoolNode::execute()
  {
    Pool kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    auto& queue = ((SYCLDevice*)getDevice())->getSYCLQueue();
    queue.parallel_for(sycl::range<2>(dst->height() * dst->numChannelBlocks(), dst->width()), [=](sycl::id<2> idx) SYCL_ESIMD_KERNEL
    {
      kernel(idx[0], idx[1]);
    });
  }

#endif

} // namespace oidn