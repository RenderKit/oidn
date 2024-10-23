// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/upsample.h"
#include "core/tensor_accessor.h"

OIDN_NAMESPACE_BEGIN

  template<typename T, TensorLayout layout>
  struct GPUUpsampleKernel
  {
    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    oidn_device_inline void operator ()(const WorkItem<3>& it) const
    {
      const int c = it.getGlobalID<0>();
      const int h = it.getGlobalID<1>();
      const int w = it.getGlobalID<2>();

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

    oidn_device_inline void operator ()(const WorkItem<3>& it) const
    {
      const int h = it.getGlobalID<0>();
      const int w = it.getGlobalID<1>();
      const int c = it.getGlobalID<2>();

      const T x = src(c, h, w);

      dst(c, h*2,   w*2)   = x;
      dst(c, h*2,   w*2+1) = x;
      dst(c, h*2+1, w*2)   = x;
      dst(c, h*2+1, w*2+1) = x;
    }
  };

  template<typename EngineT, typename SrcDstT, TensorLayout srcDstLayout>
  class GPUUpsample : public Upsample
  {
  public:
    GPUUpsample(EngineT* engine,
                const UpsampleDesc& desc)
      : Upsample(desc),
        engine(engine) {}

    Engine* getEngine() const override { return engine; }

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      if (!src || !dst)
        throw std::logic_error("upsampling source/destination not set");

      GPUUpsampleKernel<SrcDstT, srcDstLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (srcDstLayout == TensorLayout::hwc)
        engine->submitKernel(WorkDim<3>(src->getH(), src->getW(), src->getPaddedC()), kernel);
      else
        engine->submitKernel(WorkDim<3>(src->getPaddedC(), src->getH(), src->getW()), kernel);
    }

  private:
    EngineT* engine;
  };

OIDN_NAMESPACE_END
