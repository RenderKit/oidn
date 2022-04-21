// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "../tensor_accessor.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct GPUUpsampleKernel
  {
    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<3>& it) const
    {
      const int c = it.getId<0>();
      const int h = it.getId<1>();
      const int w = it.getId<2>();

      const T x = src(c, h, w);

      dst(c, h*2,   w*2)   = x;
      dst(c, h*2,   w*2+1) = x;
      dst(c, h*2+1, w*2)   = x;
      dst(c, h*2+1, w*2+1) = x;
    }
  };

  // Optimized for HWC layout (memory coalescing)
  template<typename T>
  struct GPUUpsampleKernel<T, TensorLayout::hwc>
  {
    TensorAccessor3D<T, TensorLayout::hwc> src;
    TensorAccessor3D<T, TensorLayout::hwc> dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<3>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();
      const int c = it.getId<2>();

      const T x = src(c, h, w);

      dst(c, h*2,   w*2)   = x;
      dst(c, h*2,   w*2+1) = x;
      dst(c, h*2+1, w*2)   = x;
      dst(c, h*2+1, w*2+1) = x;
    }
  };

  template<typename DeviceType, typename TensorDataType, TensorLayout tensorLayout>
  class GPUUpsample : public Upsample
  {
  public:
    GPUUpsample(const Ref<DeviceType>& device,
                const UpsampleDesc& desc)
      : Upsample(desc),
        device(device) {}

    void run() override
    {
      GPUUpsampleKernel<TensorDataType, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (tensorLayout == TensorLayout::hwc)
        device->runKernelAsync({src->getH(), src->getW(), src->getC()}, kernel);
      else
        device->runKernelAsync({src->getC(), src->getH(), src->getW()}, kernel);
    }

  private:
    Ref<DeviceType> device;
  };

} // namespace oidn
