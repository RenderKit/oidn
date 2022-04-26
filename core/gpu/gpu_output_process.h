// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../output_process.h"
#include "../tensor_accessor.h"
#include "../image_accessor.h"
#include "../color.h"
#include "../tile.h"

namespace oidn {

  template<typename ImageDataType, typename TensorDataType, TensorLayout tensorLayout>
  struct GPUOutputProcessKernel
  {
    // Source
    TensorAccessor3D<TensorDataType, tensorLayout> src;

    // Destination
    ImageAccessor<ImageDataType> dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<2>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();

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

  template<typename DeviceType, typename TensorDataType, TensorLayout tensorLayout>
  class GPUOutputProcess : public OutputProcess
  {
  public:
    GPUOutputProcess(const Ref<DeviceType>& device, const OutputProcessDesc& desc)
      : OutputProcess(desc),
        device(device) {}

    void run() override
    {
      switch (dst->getDataType())
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
      assert(tile.hSrcBegin + tile.H <= src->getH());
      assert(tile.wSrcBegin + tile.W <= src->getW());
      //assert(tile.hDstBegin + tile.H <= dst->getH());
      //assert(tile.wDstBegin + tile.W <= dst->getW());

      GPUOutputProcessKernel<ImageDataType, TensorDataType, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      device->runKernelAsync({tile.H, tile.W}, kernel);
    }

    Ref<DeviceType> device;
  };

} // namespace oidn
