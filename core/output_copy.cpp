// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(OIDN_DEVICE_GPU)
  #include "sycl_device.h"
#endif

#include "output_copy.h"
#include "output_copy_ispc.h"

namespace oidn {

  namespace
  {
    // Output copy function
    void cpuOutputCopy(const Image& src,
                       const Image& dst)
    {
      ispc::OutputCopy impl;

      impl.src = src;
      impl.dst = dst;

      impl.H = dst.height;
      impl.W = dst.width;

      parallel_nd(impl.H, [&](int h)
      {
        ispc::OutputCopy_kernel(&impl, h);
      });
    }

  #if defined(OIDN_DEVICE_GPU)

    template<typename T>
    struct OutputCopy
    {
      // Source
      ImageAccessor<T> src;

      // Destination
      ImageAccessor<T> dst;

      __forceinline void operator()(int h, int w) const
      {
        // Load
        vec3<T> value = src.get3(h, w);

        // Store
        dst.set3(h, w, value);
      }
    };

    template<typename T>
    void syclOutputCopy(const Ref<SYCLDevice>& device,
                        const Image& src,
                        const Image& dst)
    {
      OutputCopy<T> kernel;
      kernel.src = src;
      kernel.dst = dst;

      auto& queue = device->getSYCLQueue();
      queue.parallel_for(sycl::range<2>(dst.height, dst.width), [=](sycl::id<2> idx) {
        kernel(int(idx[0]), int(idx[1]));
      });
    }

  #endif
  }

  void outputCopy(const Ref<Device>& device,
                  const Image& src,
                  const Image& dst)
  {
    assert(dst.height >= src.height);
    assert(dst.width  >= src.width);

  #if defined(OIDN_DEVICE_GPU)
    if (auto syclDevice = dynamicRefCast<SYCLDevice>(device))
    {
      switch (getDataType(src.format))
      {
      case DataType::Float32: syclOutputCopy<float>(syclDevice, src, dst); break;
      case DataType::Float16: syclOutputCopy<half>(syclDevice, src, dst);  break;
      default:                assert(0);
      }
    }
    else
  #endif
      cpuOutputCopy(src, dst);
  }

} // namespace oidn