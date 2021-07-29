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

    __forceinline void operator() (int hDst, int wDst) const SYCL_ESIMD_KERNEL
    { 
      using namespace sycl::ext::intel::experimental::esimd;

      const int hSrc = hDst * 2;
      const int wSrc = wDst * 2;

      const size_t srcRowStride   = (size_t)src.W * K;
      const size_t dstRowStride   = (size_t)dst.W * K;
      const size_t srcPlaneStride = (size_t)src.H * srcRowStride;
      const size_t dstPlaneStride = (size_t)dst.H * dstRowStride;

      const size_t srcIndex = (size_t)hSrc * srcRowStride + (size_t)wSrc * K;
      const size_t dstIndex = (size_t)hDst * dstRowStride + (size_t)wDst * K;

      int16_t* srcPtr0 = (int16_t*)&src.ptr[srcIndex];
      int16_t* srcPtr1 = srcPtr0 + K;
      int16_t* srcPtr2 = srcPtr0 + srcRowStride;
      int16_t* srcPtr3 = srcPtr2 + K;
      int16_t* dstPtr  = (int16_t*)&dst.ptr[dstIndex];

      for (int c = 0; c < src.C; c += K)
      {        
        simd<int16_t, K> v0, v1, v2, v3;
        v0.copy_from(srcPtr0);
        v1.copy_from(srcPtr1);
        v2.copy_from(srcPtr2);
        v3.copy_from(srcPtr3);

        // FIXME: use half
        simd<int16_t, K> v = esimd_max(esimd_max(v0, v1), esimd_max(v2, v3));
        v.copy_to(dstPtr);

        srcPtr0 += srcPlaneStride;
        srcPtr1 += srcPlaneStride;
        srcPtr2 += srcPlaneStride;
        srcPtr3 += srcPlaneStride;
        dstPtr  += dstPlaneStride;
      }
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
    queue.parallel_for(sycl::range<2>(dst->height(), dst->width()), [=](sycl::id<2> idx) SYCL_ESIMD_KERNEL
    {
      kernel(int(idx[0]), int(idx[1]));
    });
  }

#endif

} // namespace oidn