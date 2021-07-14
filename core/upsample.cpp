// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(OIDN_DEVICE_GPU)
  #include "sycl_device.h"
#endif

#include "upsample.h"
#include "upsample_ispc.h"

namespace oidn {

#if defined(OIDN_DNNL)

  CPUUpsampleNode::CPUUpsampleNode(const Ref<Device>& device,
                                   const std::string& name,
                                   const std::shared_ptr<Tensor>& src,
                                   const std::shared_ptr<Tensor>& dst)
    : UpsampleNode(device, name, src, dst)
  {
    assert(src->layout == TensorLayout::Chw8c ||
           src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());
  }

  void CPUUpsampleNode::execute()
  {
    const int K = device->getTensorBlockSize();

    ispc::Upsample impl;
    impl.src = *src;
    impl.dst = *dst;

    parallel_nd(impl.src.C / K, impl.src.H, [&](int ck, int h)
    {
      ispc::Upsample_kernel(&impl, ck, h);
    });
  }

#else

  CPUUpsampleNode::CPUUpsampleNode(const Ref<Device>& device,
                                   const std::string& name,
                                   const std::shared_ptr<Tensor>& src,
                                   const std::shared_ptr<Tensor>& dst)
    : UpsampleNode(device, name, src, dst)
  {
    assert(src->layout == TensorLayout::chw);
  }

  void CPUUpsampleNode::execute()
  {
    const size_t C = src->dims[0];
    const size_t H = src->dims[1];
    const size_t W = src->dims[2];

    parallel_nd(C, H, [&](int c, int h)
    {
      const size_t offset = (c*H + h) * W;
      const float* srcPtr_line = (float*)src->data() + offset;
      float* dstPtr_line0 = (float*)dst->data() + offset * 4;
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

#if defined(OIDN_DEVICE_GPU)

  struct Upsample
  {
    TensorAccessor<half> src;
    TensorAccessor<half> dst;

    __forceinline void operator()(int hDst, int wDst) const
    {
      const int hSrc = hDst / 2;
      const int wSrc = wDst / 2;

      for (int c = 0; c < src.C; ++c)
      {
        dst.set(hDst, wDst, c, src.get(hSrc, wSrc, c));
      }
    }
  };

  SYCLUpsampleNode::SYCLUpsampleNode(const Ref<SYCLDevice>& device,
                                     const std::string& name,
                                     const std::shared_ptr<Tensor>& src,
                                     const std::shared_ptr<Tensor>& dst)
    : UpsampleNode(device, name, src, dst)
  {
    assert(src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());
  }

  void SYCLUpsampleNode::execute()
  {
    Upsample kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    auto queue = ((SYCLDevice*)getDevice())->getSYCLQueue();
    queue.parallel_for(sycl::range<2>(kernel.dst.H, kernel.dst.W), [=](sycl::id<2> idx) {
      kernel(int(idx[0]), int(idx[1]));
    });
  }

#endif

} // namespace oidn