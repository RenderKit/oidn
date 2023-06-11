// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/input_process.h"
#include "core/tensor_accessor.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/tile.h"

OIDN_NAMESPACE_BEGIN

  template<typename TensorDataT, TensorLayout tensorLayout, int dstPaddedC>
  struct GPUInputProcessKernel
  {
    // Source
    ImageAccessor input;  // color, albedo or normal
    ImageAccessor albedo; // auxiliary albedo
    ImageAccessor normal; // auxiliary normal

    // Destination
    TensorAccessor3D<TensorDataT, tensorLayout> dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    OIDN_DEVICE_INLINE vec3f getInput(int h, int w) const
    {
      vec3f value = input.get3(h, w);

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

      return value;
    }

    OIDN_DEVICE_INLINE vec3f getAlbedo(int h, int w) const
    {
      vec3f value = albedo.get3(h, w);

      // Sanitize
      value = math::clamp(math::nan_to_zero(value), 0.f, 1.f);

      return value;
    }

    OIDN_DEVICE_INLINE vec3f getNormal(int h, int w) const
    {
      vec3f value = normal.get3(h, w);

      // Sanitize
      value = math::clamp(math::nan_to_zero(value), -1.f, 1.f);

      // Transform to [0..1]
      value = value * 0.5f + 0.5f;

      return value;
    }

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<2>& it) const
    {
      const int hDst = it.getId<0>();
      const int wDst = it.getId<1>();

      const int h = hDst - tile.hDstBegin;
      const int w = wDst - tile.wDstBegin;

      float values[dstPaddedC] = {}; // = 0

      if (h >= 0 && h < tile.H && w >= 0 && w < tile.W)
      {
        const int hSrc = h + tile.hSrcBegin;
        const int wSrc = w + tile.wSrcBegin;

        const vec3f inputValue = getInput(hSrc, wSrc);
        values[0] = inputValue.x;
        values[1] = inputValue.y;
        values[2] = inputValue.z;

        if (dstPaddedC >= 6 && albedo.ptr)
        {
          const vec3f albedoValue = getAlbedo(hSrc, wSrc);
          values[3] = albedoValue.x;
          values[4] = albedoValue.y;
          values[5] = albedoValue.z;

          if (dstPaddedC >= 9 && normal.ptr)
          {
            const vec3f normalValue = getNormal(hSrc, wSrc);
            values[6] = normalValue.x;
            values[7] = normalValue.y;
            values[8] = normalValue.z;
          }
        }
      }

      #pragma unroll
      for (int c = 0; c < dstPaddedC; ++c)
        dst(c, hDst, wDst) = values[c];
    }
  };

  template<typename EngineT, typename TensorDataT, TensorLayout tensorLayout, int tensorBlockC>
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

      switch (dst->getC())
      {
      case  3: runImpl<3>(); break;
      case  6: runImpl<6>(); break;
      case  9: runImpl<9>(); break;
      default: throw std::logic_error("unsupported input processing source");
      }
    }

  private:
    template<int dstC>
    void runImpl()
    {
      constexpr int dstPaddedC = round_up(dstC, tensorBlockC);
      if (dst->getPaddedC() != dstPaddedC)
        throw std::logic_error("unexpected input processing destination channel count");

      using Kernel = GPUInputProcessKernel<TensorDataT, tensorLayout, dstPaddedC>;

      Kernel kernel;
      Image nullImage;

      kernel.input  = color ? *color : (albedo ? *albedo : *normal);
      kernel.albedo = (color && albedo) ? *albedo : nullImage;
      kernel.normal = (color && normal) ? *normal : nullImage;
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
