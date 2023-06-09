// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/input_process.h"
#include "core/tensor_accessor.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/tile.h"

OIDN_NAMESPACE_BEGIN

  template<typename TensorDataT, TensorLayout tensorLayout>
  struct GPUInputProcessKernel
  {
    // Source
    ImageAccessor color;
    ImageAccessor albedo;
    ImageAccessor normal;

    // Destination
    TensorAccessor3D<TensorDataT, tensorLayout> dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    OIDN_DEVICE_INLINE void storeZero(int c, int h, int w) const
    {
      dst(c, h, w) = 0.f;
    }

    // Stores a color value
    OIDN_DEVICE_INLINE void storeColor(int c, int h, int w, vec3f value) const
    {
      // Scale
      value = value * transferFunc.getInputScale();

      // Sanitize
      value = math::clamp(math::nan_to_zero(value), snorm ? -1.f : 0.f, hdr ? FLT_MAX : 1.f);

      if (snorm)
      {
        // Transform to [0..1]
        value = value * 0.5f + 0.5f;
      }

      // Apply the transfer function
      value = transferFunc.forward(value);

      // Store
      dst.set3(c, h, w, value);
    }

    // Stores an albedo value
    OIDN_DEVICE_INLINE void storeAlbedo(int c, int h, int w, vec3f value) const
    {
      // Scale
      if (!color.ptr)
        value = value * transferFunc.getInputScale();

      // Sanitize
      value = math::clamp(math::nan_to_zero(value), 0.f, 1.f);

      // Apply the transfer function
      if (!color.ptr)
        value = transferFunc.forward(value);

      // Store
      dst.set3(c, h, w, value);
    }

    // Stores a normal value
    OIDN_DEVICE_INLINE void storeNormal(int c, int h, int w, vec3f value) const
    {
      // Scale
      if (!color.ptr)
        value = value * transferFunc.getInputScale();

      // Sanitize
      value = math::clamp(math::nan_to_zero(value), -1.f, 1.f);

      // Transform to [0..1]
      value = value * 0.5f + 0.5f;

      // Store
      dst.set3(c, h, w, value);
    }

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<2>& it) const
    {
      const int hDst = it.getId<0>();
      const int wDst = it.getId<1>();

      const int h = hDst - tile.hDstBegin;
      const int w = wDst - tile.wDstBegin;

      int c = 0;

      if (h >= 0 && h < tile.H && w >= 0 && w < tile.W)
      {
        const int hSrc = h + tile.hSrcBegin;
        const int wSrc = w + tile.wSrcBegin;
        const int wDst = w + tile.wDstBegin;

        if (color.ptr)
        {
          storeColor(c, hDst, wDst, color.get3(hSrc, wSrc));
          c += 3;
        }

        if (albedo.ptr)
        {
          storeAlbedo(c, hDst, wDst, albedo.get3(hSrc, wSrc));
          c += 3;
        }

        if (normal.ptr)
        {
          storeNormal(c, hDst, wDst, normal.get3(hSrc, wSrc));
          c += 3;
        }
      }

      // Zero pad
      for (; c < dst.C; ++c)
        storeZero(c, hDst, wDst);
    }
  };

  template<typename EngineT, typename TensorDataT, TensorLayout tensorLayout>
  class GPUInputProcess : public InputProcess
  {
  public:
    GPUInputProcess(const Ref<EngineT>& engine, const InputProcessDesc& desc)
      : InputProcess(engine, desc),
        engine(engine) {}

    void submit() override
    {
      if (!getMainSrc() || !dst)
        throw std::logic_error("input processing source/destination not set");
      if (tile.hSrcBegin + tile.H > getMainSrc()->getH() ||
          tile.wSrcBegin + tile.W > getMainSrc()->getW() ||
          tile.hDstBegin + tile.H > dst->getH() ||
          tile.wDstBegin + tile.W > dst->getW())
        throw std::out_of_range("input processing source/destination out of range");

      GPUInputProcessKernel<TensorDataT, tensorLayout> kernel;
      Image nullImage;

      kernel.color  = color  ? *color  : nullImage;
      kernel.albedo = albedo ? *albedo : nullImage;
      kernel.normal = normal ? *normal : nullImage;
      kernel.dst    = *dst;
      kernel.tile   = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr   = hdr;
      kernel.snorm = snorm;

      engine->submitKernel(WorkDim<2>(dst->getH(), dst->getW()), kernel);
    }

    Ref<EngineT> engine;
  };

OIDN_NAMESPACE_END
