// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/pool.h"
#include "core/tensor_accessor.h"

OIDN_NAMESPACE_BEGIN

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

  template<typename EngineT, typename TensorDataT, TensorLayout tensorLayout>
  class GPUPool : public Pool
  {
  public:
    GPUPool(const Ref<EngineT>& engine,
            const PoolDesc& desc)
      : Pool(desc),
        engine(engine) {}

    void submit() override
    {
      if (!src || !dst)
        throw std::logic_error("pooling source/destination not set");

      GPUPoolKernel<TensorDataT, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (tensorLayout == TensorLayout::hwc)
        engine->submitKernel(WorkDim<3>(dst->getH(), dst->getW(), dst->getPaddedC()), kernel);
      else
        engine->submitKernel(WorkDim<3>(dst->getPaddedC(), dst->getH(), dst->getW()), kernel);
    }

  private:
    Ref<EngineT> engine;
  };

OIDN_NAMESPACE_END
