// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../input_process.h"
#include "../tensor_accessor.h"
#include "../image_accessor.h"
#include "../color.h"
#include "../tile.h"

namespace oidn {

  template<typename ImageDataType, typename TensorDataType, TensorLayout tensorLayout>
  struct GPUInputProcessKernel
  {
    // Source
    ImageAccessor<ImageDataType> color;
    ImageAccessor<ImageDataType> albedo;
    ImageAccessor<ImageDataType> normal;

    // Destination
    TensorAccessor3D<TensorDataType, tensorLayout> dst;

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
      value = clamp(nan_to_zero(value), snorm ? -1.f : 0.f, hdr ? FLT_MAX : 1.f);

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
      value = clamp(nan_to_zero(value), 0.f, 1.f);

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
      value = clamp(nan_to_zero(value), -1.f, 1.f);

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

      if (h >= 0 && h < tile.H && w >= 0 && w < tile.W)
      {
        const int hSrc = h + tile.hSrcBegin;
        const int wSrc = w + tile.wSrcBegin;
        const int wDst = w + tile.wDstBegin;

        int c = 0;

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

        for (; c < dst.C; ++c)
          storeZero(c, hDst, wDst);
      }
      else
      {
        // Zero pad
        for (int c = 0; c < dst.C; ++c)
          storeZero(c, hDst, wDst);
      }
    }
  };

  template<typename DeviceType, typename TensorDataType, TensorLayout tensorLayout>
  class GPUInputProcess : public InputProcess
  {
  public:
    GPUInputProcess(const Ref<DeviceType>& device, const InputProcessDesc& desc)
      : InputProcess(device, desc),
        device(device) {}

    void run() override
    {
      switch (getInput()->getDataType())
      {
      case DataType::Float32: runImpl<float>(); break;
      case DataType::Float16: runImpl<half>();  break;
      default:                assert(0);
      }
    }

  private:
    template<typename ImageDataType>
    void runImpl()
    {
      assert(tile.H + tile.hSrcBegin <= getInput()->getH());
      assert(tile.W + tile.wSrcBegin <= getInput()->getW());
      assert(tile.H + tile.hDstBegin <= dst->getH());
      assert(tile.W + tile.wDstBegin <= dst->getW());
      
      GPUInputProcessKernel<ImageDataType, TensorDataType, tensorLayout> kernel;
      kernel.color  = color  ? *color  : Image();
      kernel.albedo = albedo ? *albedo : Image();
      kernel.normal = normal ? *normal : Image();
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      device->runKernelAsync({dst->getH(), dst->getW()}, kernel);
    }

    Ref<DeviceType> device;
  };

} // namespace oidn
