// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/kernel.h"
#include "core/image_accessor.h"

#if !defined(OIDN_COMPILE_METAL_DEVICE)
  #include "core/image_copy.h"
#endif

OIDN_NAMESPACE_BEGIN

  struct GPUImageCopyKernel
  {
    ImageAccessor src;
    ImageAccessor dst;

    oidn_device_inline void operator ()(const oidn_private WorkItem<2>& it) const
    {
      const int h = it.getGlobalID<0>();
      const int w = it.getGlobalID<1>();
      const vec3f value = src.get3(h, w);
      dst.set3(h, w, value);
    }
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE)

  template<typename EngineT>
  class GPUImageCopy final : public ImageCopy
  {
  public:
    explicit GPUImageCopy(EngineT* engine)
      : engine(engine) {}

    Engine* getEngine() const override { return engine; }

  #if defined(OIDN_COMPILE_METAL)
    void finalize() override
    {
      pipeline = engine->newPipeline("imageCopy");
    }
  #endif

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      check();

      GPUImageCopyKernel kernel;
      kernel.src = *src;
      kernel.dst = *dst;

    #if defined(OIDN_COMPILE_METAL)
      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel,
                           pipeline, {src->getBuffer(), dst->getBuffer()});
    #else
      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel);
    #endif
    }

  private:
    EngineT* engine;

  #if defined(OIDN_COMPILE_METAL)
    Ref<MetalPipeline> pipeline;
  #endif
  };

#endif // !defined(OIDN_COMPILE_METAL_DEVICE)

OIDN_NAMESPACE_END
