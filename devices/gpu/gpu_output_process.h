// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/kernel.h"
#include "core/tensor_accessor.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/tile.h"

#if !defined(OIDN_COMPILE_METAL_DEVICE)
  #include "core/output_process.h"
#endif

OIDN_NAMESPACE_BEGIN

  template<typename SrcT, TensorLayout srcLayout>
  struct GPUOutputProcessKernel
  {
    // Source
    TensorAccessor3D<SrcT, srcLayout> src;

    // Destination
    ImageAccessor dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    oidn_device_inline void operator ()(const oidn_private WorkItem<2>& it) const
    {
      const int h = it.getGlobalID<0>();
      const int w = it.getGlobalID<1>();

      const int hSrc = h + tile.hSrcBegin;
      const int hDst = h + tile.hDstBegin;
      const int wSrc = w + tile.wSrcBegin;
      const int wDst = w + tile.wDstBegin;

      // Load
      vec3f value = src.get3(0, hSrc, wSrc);

      // The CNN output may contain negative values or even NaNs, so it must be sanitized
      value = math::clamp(math::nan_to_zero(value), 0.f, FLT_MAX);

      // Apply the inverse transfer function
      value = transferFunc.inverse(value);

      // Average the channels if there is only one output channel
      if (dst.C == 1)
        value = (value.x + value.y + value.z) * (1.f / 3.f);

      // Sanitize
      if (snorm)
      {
        // Transform to [-1..1]
        value = value * 2.f - 1.f;
        value = math::max(value, -1.f);
      }
      if (!hdr)
        value = math::min(value, 1.f);

      // Scale
      value = value * transferFunc.getOutputScale();

      // Store
      dst.set3(hDst, wDst, value);
    }
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE)

  template<typename EngineT, typename SrcT, TensorLayout srcLayout>
  class GPUOutputProcess : public OutputProcess
  {
  public:
    GPUOutputProcess(EngineT* engine, const OutputProcessDesc& desc)
      : OutputProcess(desc),
        engine(engine) {}

    Engine* getEngine() const override { return engine; }

  #if defined(OIDN_COMPILE_METAL)
    void setScratch(const Ref<Buffer>& scratch) override
    {
      this->scratch = scratch;
    }

    void finalize() override
    {
      const std::string kernelName = "outputProcess_" + toString(DataTypeOf<SrcT>::value) +
                                     "_" + toString(srcLayout);
      pipeline = engine->newPipeline(kernelName);
    }
  #endif

    void submitKernels(const Ref<CancellationToken>& ct) override
    {
      check();

      GPUOutputProcessKernel<SrcT, srcLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

    #if defined(OIDN_COMPILE_METAL)
      engine->submitKernel(WorkDim<2>(tile.H, tile.W), kernel,
                           pipeline, {src->getBuffer(), dst->getBuffer(), scratch});
    #else
      engine->submitKernel(WorkDim<2>(tile.H, tile.W), kernel);
    #endif
    }

  private:
    EngineT* engine;

  #if defined(OIDN_COMPILE_METAL)
    Ref<MetalPipeline> pipeline;
    Ref<Buffer> scratch; // may contain autoexposure result, which must be tracked for Metal
  #endif
  };

#endif // !defined(OIDN_COMPILE_METAL_DEVICE)

OIDN_NAMESPACE_END
