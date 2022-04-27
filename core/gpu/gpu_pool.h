// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "../tensor_accessor.h"

namespace oidn {

  template<typename T, TensorLayout layout>
  struct GPUPoolKernel
  {
    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<3>& it) const
    {
      const int c = it.getId<0>();
      const int h = it.getId<1>();
      const int w = it.getId<2>();

      const T x0 = src(c, h*2,   w*2);
      const T x1 = src(c, h*2,   w*2+1);
      const T x2 = src(c, h*2+1, w*2);
      const T x3 = src(c, h*2+1, w*2+1);

      dst(c, h, w) = math::max(math::max(x0, x1), math::max(x2, x3));
    }
  };

  // Optimized for HWC layout (memory coalescing)
  template<typename T>
  struct GPUPoolKernel<T, TensorLayout::hwc>
  {
    TensorAccessor3D<T, TensorLayout::hwc> src;
    TensorAccessor3D<T, TensorLayout::hwc> dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<3>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();
      const int c = it.getId<2>();

      const T x0 = src(c, h*2,   w*2);
      const T x1 = src(c, h*2,   w*2+1);
      const T x2 = src(c, h*2+1, w*2);
      const T x3 = src(c, h*2+1, w*2+1);

      dst(c, h, w) = math::max(math::max(x0, x1), math::max(x2, x3));
    }
  };

  template<typename DeviceType, typename TensorDataType, TensorLayout tensorLayout>
  class GPUPool : public Pool
  {
  public:
    GPUPool(const Ref<DeviceType>& device,
            const PoolDesc& desc)
      : Pool(desc),
        device(device) {}

    void run() override
    {
      GPUPoolKernel<TensorDataType, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (tensorLayout == TensorLayout::hwc)
        device->runKernelAsync({dst->getH(), dst->getW(), dst->getC()}, kernel);
      else
        device->runKernelAsync({dst->getC(), dst->getH(), dst->getW()}, kernel);
    }

  private:
    Ref<DeviceType> device;
  };

} // namespace oidn
